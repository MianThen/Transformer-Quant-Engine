from __future__ import annotations

import dataclasses
import numpy as np
import pandas as pd
import pytest
import json

from python.qbt_ml.labels import (
    LabelSpecV2,
    RankingScoreMode,
    RankingScoreSpecV1,
    build_label_v2,
    production_ranking_score,
)
from python.qbt_ml.training import (
    CrossSectionBatchSampler,
    KENDALL_TASKS,
    KendallTaskWeights,
    compare_ranking_variants_numpy,
    lambda_loss_at_k,
    lambda_loss_numpy,
    listmle_loss,
    listmle_loss_numpy,
    multitask_loss_components,
    ndcg_at_k,
    ranking_oos_metrics,
    run_phase1b_ablation,
    timestamp_walk_forward_splits,
)
from python.qbt_ml.export.manifest import ModelManifest
from python.qbt_ml.cli import _build_dataset, _train


def _label_bars(count: int = 10) -> pd.DataFrame:
    rows = []
    for timestamp in range(1, count + 1):
        for symbol, slope in (("A", 0.08), ("B", -0.03), ("C", 0.08)):
            close = 10.0 + slope * timestamp
            rows.append({
                "timestamp": timestamp,
                "symbol": symbol,
                "open": close - slope * 0.4,
                "close": close,
            })
    return pd.DataFrame(rows)


def test_label_spec_v2_alignment_soft_risk_and_cross_section_relevance():
    ranking = RankingScoreSpecV1(
        production_top_k=2,
        risk_floor=0.01,
        winsor_lower_quantile=0.0,
        winsor_upper_quantile=1.0,
    )
    spec = LabelSpecV2(
        horizon_bars=2,
        direction_temperature=0.01,
        ranking=ranking,
    )
    bars = _label_bars()
    labels = build_label_v2(bars.sample(frac=1.0, random_state=4), spec)
    first = labels[(labels.timestamp == 1) & (labels.symbol == "A")].iloc[0]
    path = bars[bars.symbol == "A"].sort_values("timestamp")
    assert first.entry_open == pytest.approx(path.iloc[1].open)
    assert first.exit_close == pytest.approx(path.iloc[3].close)
    assert first.return_raw == pytest.approx(np.log(first.exit_close / first.entry_open))
    assert first.expected_return == first.return_raw
    assert 0.5 < first.direction_soft < 1.0
    assert first.realized_volatility >= 0.0
    assert first.downside_semivol >= 0.0
    assert first.risk_adjusted_return == pytest.approx(
        first.return_raw / max(first.realized_volatility, ranking.risk_floor)
    )
    cross_section = labels[labels.timestamp == 1].set_index("symbol")
    assert cross_section.loc["A", "rank_relevance"] == pytest.approx(
        cross_section.loc["C", "rank_relevance"]
    )
    assert cross_section.loc["A", "rank_relevance"] > cross_section.loc["B", "rank_relevance"]
    assert labels.label_spec_sha256.nunique() == 1
    assert labels.ranking_score_spec_sha256.nunique() == 1


def test_label_spec_v2_future_mutation_does_not_change_earlier_labels():
    spec = LabelSpecV2(
        horizon_bars=2,
        ranking=RankingScoreSpecV1(
            production_top_k=2,
            winsor_lower_quantile=0.0,
            winsor_upper_quantile=1.0,
        ),
    )
    source = _label_bars()
    before = build_label_v2(source, spec)
    changed = source.copy()
    changed.loc[changed.timestamp >= 8, ["open", "close"]] *= 3.0
    after = build_label_v2(changed, spec)
    columns = [
        "return_raw", "direction_soft", "realized_volatility",
        "downside_semivol", "risk_adjusted_return", "rank_utility",
        "rank_relevance", "label_valid",
    ]
    cutoff = before.timestamp <= 4
    pd.testing.assert_frame_equal(
        before.loc[cutoff, columns].reset_index(drop=True),
        after.loc[cutoff, columns].reset_index(drop=True),
    )


def test_ranking_score_spec_raw_and_risk_adjusted_contract():
    expected = np.asarray([0.02, 0.01, -0.01])
    volatility = np.asarray([0.10, 0.02, 0.20])
    raw = RankingScoreSpecV1(mode=RankingScoreMode.RAW_RETURN, production_top_k=2)
    adjusted = RankingScoreSpecV1(
        mode=RankingScoreMode.RISK_ADJUSTED_RETURN,
        production_top_k=2,
        risk_floor=0.05,
    )
    np.testing.assert_array_equal(
        production_ranking_score(expected, volatility, raw), expected
    )
    np.testing.assert_allclose(
        production_ranking_score(expected, volatility, adjusted),
        expected / np.maximum(volatility, 0.05),
    )
    assert raw.sha256 != adjusted.sha256


def test_cross_section_sampler_never_splits_a_timestamp_and_replays_seed():
    timestamps = np.asarray([3, 1, 2, 1, 3, 3, 2])
    first = CrossSectionBatchSampler(
        timestamps, cross_sections_per_batch=2, shuffle=True, seed=17
    )
    second = CrossSectionBatchSampler(
        timestamps, cross_sections_per_batch=2, shuffle=True, seed=17
    )
    first_batches = list(first)
    assert first_batches == list(second)
    assert sorted(index for batch in first_batches for index in batch) == list(range(7))
    batch_by_index = {
        index: batch_number
        for batch_number, batch in enumerate(first_batches)
        for index in batch
    }
    for timestamp in np.unique(timestamps):
        assert len({batch_by_index[index] for index in np.flatnonzero(timestamps == timestamp)}) == 1


def test_listmle_numpy_is_permutation_equivalent_with_symbol_tie_policy():
    scores = np.asarray([0.3, -0.1, 0.2, 0.0])
    utility = np.asarray([1.0, 0.0, 1.0, 0.5])
    symbols = np.asarray(["B", "D", "A", "C"])
    expected = listmle_loss_numpy(scores, utility, tie_breaker=symbols)
    permutation = np.asarray([2, 0, 3, 1])
    actual = listmle_loss_numpy(
        scores[permutation], utility[permutation], tie_breaker=symbols[permutation]
    )
    assert actual == pytest.approx(expected, abs=1e-14)
    shifted = listmle_loss_numpy(scores + 100.0, utility, tie_breaker=symbols)
    assert shifted == pytest.approx(expected, abs=1e-12)
    with pytest.raises(ValueError, match="tie_breaker"):
        listmle_loss_numpy(scores, utility)


def test_ndcg_and_lambdaloss_optimized_path_match_full_pair_oracle():
    scores = np.asarray([0.7, 0.2, -0.1, 0.4, 0.0])
    relevance = np.asarray([1.0, 0.25, 0.0, 0.75, 0.5])
    optimized, optimized_diagnostics = lambda_loss_numpy(
        scores, relevance, cutoff=2, temperature=0.7, optimized_topk=True
    )
    full, full_diagnostics = lambda_loss_numpy(
        scores, relevance, cutoff=2, temperature=0.7, optimized_topk=False
    )
    assert optimized == pytest.approx(full, abs=1e-14)
    assert optimized_diagnostics.pair_count == full_diagnostics.pair_count
    assert lambda_loss_numpy(
        scores + 123.0, relevance, cutoff=2, temperature=0.7
    )[0] == pytest.approx(optimized, abs=1e-13)
    assert ndcg_at_k(relevance, relevance, cutoff=2) == pytest.approx(1.0)

    better_margin = scores.copy()
    better_margin[np.argmax(relevance)] += 1.0
    better = lambda_loss_numpy(
        better_margin, relevance, cutoff=2, temperature=0.7
    )[0]
    assert better < optimized


def test_lambdaloss_swap_weight_matches_bruteforce_delta_ndcg():
    relevance = np.asarray([1.0, 0.6, 0.1])
    scores = np.asarray([0.9, 0.5, 0.0])
    cutoff = 2
    base = ndcg_at_k(scores, relevance, cutoff=cutoff)
    swapped_scores = scores.copy()
    swapped_scores[[0, 2]] = swapped_scores[[2, 0]]
    swapped = ndcg_at_k(swapped_scores, relevance, cutoff=cutoff)
    gain = np.exp2(relevance) - 1.0
    discounts = np.asarray([1.0, 1.0 / np.log2(3.0), 0.0])
    ideal_dcg = float(np.dot(np.sort(gain)[::-1], discounts))
    formula = abs(gain[0] - gain[2]) * abs(discounts[0] - discounts[2]) / ideal_dcg
    assert abs(base - swapped) == pytest.approx(formula, abs=1e-14)


def test_lambdaloss_ties_masks_and_zero_idcg_are_finite_and_invariant():
    scores = np.asarray([0.0, 0.0, 0.0, 0.0])
    relevance = np.asarray([0.5, 0.5, 0.5, 0.5])
    loss, diagnostics = lambda_loss_numpy(scores, relevance, cutoff=2)
    assert loss == 0.0 and diagnostics.pair_count == 0
    zero_loss, zero_diagnostics = lambda_loss_numpy(
        scores, np.zeros_like(relevance), cutoff=2
    )
    assert zero_loss == 0.0 and zero_diagnostics.idcg_zero_cross_sections == 1
    mask = np.asarray([True, False, True, True])
    masked = lambda_loss_numpy(scores, relevance, cutoff=2, mask=mask)[0]
    permutation = np.asarray([2, 0, 3, 1])
    permuted = lambda_loss_numpy(
        scores[permutation], relevance[permutation], cutoff=2, mask=mask[permutation]
    )[0]
    assert masked == pytest.approx(permuted, abs=1e-14)


def test_optional_torch_losses_have_finite_gradients_and_match_numpy():
    torch = pytest.importorskip("torch")
    scores = torch.tensor([0.7, 0.2, -0.1, 0.4], dtype=torch.float64, requires_grad=True)
    relevance = torch.tensor([1.0, 0.25, 0.0, 0.75], dtype=torch.float64)
    loss = lambda_loss_at_k(scores, relevance, cutoff=2, temperature=0.7)
    expected = lambda_loss_numpy(
        scores.detach().numpy(), relevance.numpy(), cutoff=2, temperature=0.7
    )[0]
    assert float(loss.detach()) == pytest.approx(expected, abs=1e-14)
    loss.backward()
    assert torch.isfinite(scores.grad).all()

    list_scores = torch.tensor([0.3, -0.1, 0.2], dtype=torch.float64, requires_grad=True)
    utility = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
    symbols = torch.tensor([2, 3, 1])
    list_loss = listmle_loss(list_scores, utility, tie_breaker=symbols)
    list_loss.backward()
    assert torch.isfinite(list_scores.grad).all()


def test_phase1b_manifest_freezes_label_score_loss_cutoff_and_temperature():
    ranking = RankingScoreSpecV1(production_top_k=3, rank_temperature=0.7)
    label = LabelSpecV2(horizon_bars=2, ranking=ranking)
    manifest = ModelManifest(
        schema_version=1,
        model_id="phase1b-test",
        model_version="1",
        model_sha256="a" * 64,
        feature_profile="BAR_V1",
        feature_schema_sha256="b" * 64,
        calendar_id="calendar",
        universe_id="universe",
        data_cutoff_utc="2026-07-30T00:00:00Z",
        lookback=8,
        feature_count=4,
        static_feature_count=0,
        outputs=tuple({
            "name": name,
            "unit": "log_return",
            "horizon_bars": 2,
        } for name in (
            "expected_return", "expected_volatility", "direction_probability",
            "lower_quantile", "upper_quantile", "confidence",
        )),
        label_spec_version="V2",
        label_spec_sha256=label.sha256,
        ranking_score_spec={
            **ranking.__dict__, "mode": ranking.mode.value,
        },
        ranking_score_spec_sha256=ranking.sha256,
        ranking_loss_variant="lambda",
        ranking_cutoff=ranking.production_top_k,
        ranking_temperature=ranking.rank_temperature,
        rank_weight=ranking.lambda_rank,
    )
    manifest.validate()
    assert manifest.ranking_cutoff == 3
    assert manifest.ranking_temperature == pytest.approx(0.7)


def test_fixed_input_ranking_variant_report_is_timestamp_equal_weighted_and_replayable():
    scores = np.asarray([0.4, 0.1, 0.3, 0.2, -0.1])
    utility = np.asarray([1.0, 0.0, 0.8, 0.4, 0.1])
    relevance = np.asarray([1.0, 0.0, 1.0, 0.5, 0.0])
    timestamps = np.asarray([1, 1, 2, 2, 2])
    symbols = np.asarray(["A", "B", "A", "B", "C"])
    first = compare_ranking_variants_numpy(
        scores, utility, relevance, timestamps, symbols,
        cutoff=2, temperature=0.7,
    )
    second = compare_ranking_variants_numpy(
        scores, utility, relevance, timestamps, symbols,
        cutoff=2, temperature=0.7,
    )
    assert first == second
    assert first.cross_sections == 2
    assert len(first.input_fingerprint) == 64
    assert np.isfinite([
        first.legacy_loss, first.listmle_loss, first.lambda_loss,
        first.ndcg_at_cutoff,
    ]).all()


def test_oos_ranking_metrics_freeze_topk_ties_and_turnover_definition():
    metrics = ranking_oos_metrics(
        scores=np.asarray([0.9, 0.8, 0.1, 0.7, 0.6, 0.2]),
        utility=np.asarray([1.0, 0.5, 0.0, 0.0, 1.0, 0.5]),
        relevance=np.asarray([1.0, 0.5, 0.0, 0.0, 1.0, 0.5]),
        timestamps=np.asarray([1, 1, 1, 2, 2, 2]),
        symbols=np.asarray(["A", "B", "C", "A", "B", "C"]),
        cutoff=2,
    )
    assert metrics.cross_sections == 2
    assert metrics.transition_count == 1
    assert metrics.precision_at_cutoff == pytest.approx(0.75)
    assert metrics.top_k_overlap == pytest.approx(1.0)
    assert metrics.top_k_turnover == pytest.approx(0.0)
    assert np.isfinite(dataclasses.astuple(metrics)).all()


def test_timestamp_walk_forward_keeps_three_purged_atomic_oos_windows():
    timestamps = np.repeat(np.arange(45), 3)
    folds = list(timestamp_walk_forward_splits(
        timestamps, train_size=10, validation_size=4, test_size=6,
        step=10, purge_timestamps=2, embargo_timestamps=1,
    ))
    assert len(folds) == 3
    for fold in folds:
        assert set(timestamps[fold.test]) == set(fold.test_timestamps)
        assert fold.train_timestamps[-1] < fold.validation_timestamps[0]
        assert fold.validation_timestamps[-1] < fold.test_timestamps[0]


def test_phase1b_ablation_runner_freezes_contract_and_requires_economic_gate(tmp_path):
    dataset = tmp_path / "dataset.npz"
    timestamps = np.repeat(np.arange(45), 2)
    np.savez_compressed(
        dataset,
        timestamps=timestamps,
        label_spec_sha256=np.asarray("a" * 64),
        ranking_score_spec_sha256=np.asarray("b" * 64),
    )
    config = {
        "enabled": True,
        "seed": 17,
        "ranking": {"loss_variant": "legacy"},
        "training": {"epochs": 1, "output": "ignored"},
        "phase1b_ablation": {
            "enabled": True,
            "variants": ["legacy", "listmle", "lambda"],
            "train_timestamps": 10,
            "validation_timestamps": 4,
            "test_timestamps": 6,
            "step_timestamps": 10,
            "purge_timestamps": 2,
            "embargo_timestamps": 1,
            "window_count": 3,
        },
    }

    def fake_train(run_config, _dataset, output, *, _split_override):
        variant = run_config["ranking"]["loss_variant"]
        adjustment = {"legacy": 0.0, "listmle": -0.01, "lambda": 0.02}[variant]
        path = __import__("pathlib").Path(output)
        path.mkdir(parents=True)
        (path / "metrics.json").write_text(json.dumps({
            "test_ranking": {
                "ndcg_at_cutoff": 0.7 + adjustment,
                "rank_ic": 0.2 + adjustment,
                "precision_at_cutoff": 0.4 + adjustment,
                "top_k_overlap": 0.8,
                "top_k_turnover": 0.2,
                "top_bottom_utility_spread": 0.01 + max(adjustment, 0.0),
            }
        }), encoding="utf-8")

    output = tmp_path / "ablation"
    run_phase1b_ablation(config, dataset, output, fake_train)
    report = json.loads((output / "paired_report.json").read_text(encoding="utf-8"))
    assert report["contract"]["window_count"] == 3
    assert report["promotion"]["model_metric_winner"] == "lambda"
    assert report["promotion"]["winner_frozen"] is False
    assert report["promotion"]["status"] == "awaiting_cpp_economic_gate"


def test_fixed_weight_multitask_lambda_components_are_finite_and_backpropagate():
    torch = pytest.importorskip("torch")
    expected_return = torch.tensor(
        [0.03, 0.01, -0.01, 0.02], dtype=torch.float64, requires_grad=True
    )
    expected_volatility = torch.tensor(
        [0.10, 0.08, 0.12, 0.09], dtype=torch.float64, requires_grad=True
    )
    prediction = {
        "expected_return": expected_return,
        "expected_volatility": expected_volatility,
        "direction_probability": torch.tensor(
            [0.7, 0.6, 0.3, 0.65], dtype=torch.float64, requires_grad=True
        ),
        "lower_quantile": torch.tensor(
            [-0.02, -0.01, -0.04, -0.01], dtype=torch.float64, requires_grad=True
        ),
        "upper_quantile": torch.tensor(
            [0.05, 0.03, 0.01, 0.04], dtype=torch.float64, requires_grad=True
        ),
    }
    target = {
        "expected_return": torch.tensor([0.04, 0.0, -0.02, 0.03], dtype=torch.float64),
        "direction": torch.tensor([0.9, 0.5, 0.1, 0.8], dtype=torch.float64),
        "realized_volatility": torch.tensor([0.11, 0.07, 0.13, 0.10], dtype=torch.float64),
        "rank_utility": torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float64),
        "rank_relevance": torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float64),
        "timestamp": torch.tensor([1, 1, 2, 2]),
        "symbol_tie_breaker": torch.tensor([1, 2, 1, 2]),
    }
    components = multitask_loss_components(
        prediction,
        target,
        ranking_variant="lambda",
        ranking_cutoff=1,
        rank_temperature=0.7,
        ranking_score_mode="raw_return",
        risk_floor=1e-4,
    )
    assert set(components) == {"return", "direction", "volatility", "quantile", "rank"}
    total = sum(components.values())
    assert torch.isfinite(total)
    total.backward()
    assert expected_return.grad is not None and torch.isfinite(expected_return.grad).all()
    assert expected_volatility.grad is not None and torch.isfinite(expected_volatility.grad).all()


def test_kendall_task_weights_keep_rank_explicit_and_record_finite_gradients():
    torch = pytest.importorskip("torch")
    module = KendallTaskWeights(
        initial_log_variance=0.0, regularizer=0.5, minimum=-2.0, maximum=2.0
    )
    components = {
        task: torch.tensor(float(index + 1), requires_grad=True)
        for index, task in enumerate(KENDALL_TASKS)
    }
    total, weighted = module(components)
    assert float(total.detach()) == pytest.approx(10.0)
    assert set(weighted) == set(KENDALL_TASKS)
    total.backward()
    for task in KENDALL_TASKS:
        assert torch.isfinite(module.log_variances[task].grad)
    with torch.no_grad():
        module.log_variances["return"].fill_(100.0)
    module.clamp_()
    assert float(module.log_variances["return"].detach()) == pytest.approx(2.0)


def test_phase1b_lambda_training_smoke_writes_frozen_metrics_and_checkpoint(tmp_path):
    pytest.importorskip("torch")
    rows = []
    for timestamp in range(1, 91):
        for offset, symbol in enumerate(("A", "B", "C")):
            close = 10.0 + offset + 0.02 * timestamp + 0.03 * np.sin(
                timestamp / (3.0 + offset)
            )
            rows.append({
                "timestamp": timestamp,
                "symbol": symbol,
                "open": close * (0.999 + offset * 0.0001),
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1000 + timestamp * (offset + 1),
                "is_listed": True,
                "is_suspended": False,
                "is_st": False,
            })
    source = tmp_path / "bars.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    dataset = tmp_path / "dataset.npz"
    run = tmp_path / "run"
    config = {
        "enabled": True,
        "seed": 20260730,
        "lookback": 8,
        "label_horizon_bars": 1,
        "data": {"source": str(source), "dataset_output": str(dataset)},
        "label_v2": {
            "horizon_bars": 1,
            "execution_lag_bars": 1,
            "direction_threshold": 0.0,
            "direction_temperature": 0.0025,
        },
        "ranking_score": {
            "mode": "raw_return",
            "production_top_k": 2,
            "risk_floor": 1e-4,
            "cost_proxy_bps": 0.0,
            "winsor_lower_quantile": 0.0,
            "winsor_upper_quantile": 1.0,
            "rank_temperature": 0.7,
            "lambda_rank": 0.1,
            "target_tie_policy": "symbol_ascending",
        },
        "ranking": {
            "loss_variant": "lambda",
            "cross_sections_per_batch": 1,
            "return_weight": 1.0,
            "direction_weight": 0.25,
            "volatility_weight": 0.25,
            "quantile_weight": 0.25,
        },
        "split": {
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "purge_timestamps": 2,
            "embargo_timestamps": 1,
        },
        "model": {
            "d_model": 8,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 16,
            "dropout": 0.0,
        },
        "training": {
            "dataset": str(dataset),
            "output": str(run),
            "epochs": 1,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "device": "cpu",
        },
    }
    _build_dataset(config, None)
    _train(config, None, None)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert np.isfinite(metrics["validation_ndcg_at_k"])
    assert np.isfinite(metrics["test_ndcg_at_k"])
    assert metrics["validation_cross_sections"] > 0
    checkpoint = __import__("torch").load(
        run / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["ranking_loss_variant"] == "lambda"
    assert checkpoint["ranking_cutoff"] == 2
    assert len(checkpoint["label_spec_sha256"]) == 64
    assert len(checkpoint["ranking_score_spec_sha256"]) == 64

    dynamic_run = tmp_path / "dynamic-run"
    dynamic_config = json.loads(json.dumps(config))
    dynamic_config["training"]["output"] = str(dynamic_run)
    dynamic_config["multitask_weighting"] = {
        "mode": "kendall",
        "frozen_ranking_loss": "lambda",
        "initial_log_variance": 0.0,
        "regularizer": 0.5,
        "minimum_log_variance": -6.0,
        "maximum_log_variance": 6.0,
    }
    _train(dynamic_config, None, None)
    dynamic_checkpoint = __import__("torch").load(
        dynamic_run / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    assert dynamic_checkpoint["multitask_weighting_mode"] == "kendall"
    diagnostics = dynamic_checkpoint["loss_weight_diagnostics"]
    assert len(diagnostics) == 1
    assert set(diagnostics[0]["log_variances"]) == set(KENDALL_TASKS)
    assert np.isfinite([
        value
        for kind in ("raw_losses", "weighted_losses", "log_variance_gradients",
                     "log_variances", "effective_weights")
        for value in diagnostics[0][kind].values()
    ]).all()
