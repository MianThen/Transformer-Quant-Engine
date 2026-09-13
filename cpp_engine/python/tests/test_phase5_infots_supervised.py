"""Phase 5 supervised ablation trainer tests (CPU, tiny synthetic dataset)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from python.qbt_ml.evaluation.infots_ablation import (  # noqa: E402
    InfoTSBudgetContractV1,
    validate_infots_fold_artifact,
)
from python.qbt_ml.research.infots import InfoTSAugmentationSpecV1  # noqa: E402
from python.qbt_ml.research.infots_supervised import (  # noqa: E402
    Phase5GateSpecV1,
    Phase5SupervisedSpecV1,
    _spearman,
    build_stress_features,
    compute_supervised_metrics,
    load_encoder_state,
    train_phase5_supervised_fold,
)
from python.qbt_ml.research.infots_training import (  # noqa: E402
    InfoTSTrainingSpecV1,
    train_infots_pretraining,
)

TIMESTAMP_STEP = 60_000_000_000
FOLD = {
    "fold": 1,
    "train_first": 1000,
    "train_end": 1000 + 40 * TIMESTAMP_STEP,
    "validation_first": 1000 + 41 * TIMESTAMP_STEP,
    "validation_end": 1000 + 48 * TIMESTAMP_STEP,
    "test_start": 1000 + 50 * TIMESTAMP_STEP,
    "test_end": 1000 + 63 * TIMESTAMP_STEP,
    "purge_gap": TIMESTAMP_STEP,
}


def _build_dataset(seed: int = 7, symbols_per_ts: int = 5, timestamps_count: int = 64) -> dict:
    generator = np.random.default_rng(seed)
    timestamps = (1000 + np.arange(timestamps_count) * TIMESTAMP_STEP).repeat(symbols_per_ts)
    rows = timestamps.size
    features = generator.normal(0.0, 0.05, size=(rows, 8, 23)).astype(np.float32)
    features[:, :, 19:] = generator.integers(0, 2, size=(rows, 8, 4)).astype(np.float32)
    valid_mask = np.ones((rows, 8), dtype=np.uint8)
    symbols = np.array([f"{index % symbols_per_ts:06d}" for index in range(rows)])
    expected_return = generator.normal(0.0, 0.02, size=rows).astype(np.float32)
    direction = (0.5 + 20.0 * expected_return).clip(0.05, 0.95).astype(np.float32)
    realized_volatility = np.abs(generator.normal(0.0, 0.01, size=rows)).astype(np.float32) + 1e-4
    rank_relevance = (expected_return - expected_return.min()) / (np.ptp(expected_return) + 1e-9)
    rank_utility = expected_return / (realized_volatility + 1e-6)
    return {
        "features": features,
        "valid_mask": valid_mask,
        "timestamps": timestamps,
        "symbols": symbols,
        "expected_return": expected_return,
        "direction": direction,
        "realized_volatility": realized_volatility,
        "rank_relevance": rank_relevance.astype(np.float32),
        "rank_utility": rank_utility.astype(np.float32),
    }


def _specs(**overrides):
    settings = dict(
        seed=20260911, epochs=1, batch_size=32, hidden_dim=16, embedding_dim=8,
        learning_rate=1e-3, device="cpu",
    )
    settings.update(overrides)
    return Phase5SupervisedSpecV1(**settings)


def _augmentation(augmentation_id: str) -> InfoTSAugmentationSpecV1:
    return InfoTSAugmentationSpecV1(
        schema_version=1,
        augmentation_id=augmentation_id,
        seed=20260911,
        crop_min_length=4,
        crop_max_length=8,
        noise_std=0.01,
        time_mask_probability=0.0,
        continuous_feature_indices=[],
        state_feature_indices=[19, 20, 21, 22],
        padding_value=0.0,
        future_guard=True,
    )


def _pretrain_checkpoint(dataset: dict, tmp_path: Path, train_fold_end: int) -> Path:
    rows = np.flatnonzero(dataset["timestamps"] <= train_fold_end)
    spec = InfoTSTrainingSpecV1(
        seed=20260911, epochs=1, batch_size=32, hidden_dim=16, embedding_dim=8,
        train_fold_end=train_fold_end, device="cpu",
    )
    train_infots_pretraining(
        dataset["features"][rows], dataset["valid_mask"][rows], dataset["timestamps"][rows],
        _augmentation("TEST-PRETRAIN-V1"), spec,
        checkpoint_path=tmp_path / "pretrain.pt",
    )
    return tmp_path / "pretrain.pt"


def _contract(gate_sha: str, budget: int = 1) -> InfoTSBudgetContractV1:
    return InfoTSBudgetContractV1(gate_spec_sha256=gate_sha, supervised_budget=budget)


def test_spearman_and_metrics():
    assert _spearman(np.array([1.0, 2, 3]), np.array([3.0, 2, 1])) == -1.0
    dataset = _build_dataset()
    from python.qbt_ml.research.infots_supervised import _timestamp_groups
    groups = _timestamp_groups(dataset["timestamps"])
    prediction = {
        "expected_return": dataset["expected_return"],
        "expected_volatility": dataset["realized_volatility"],
        "direction_probability": dataset["direction"],
        "lower_quantile": dataset["expected_return"] - 0.05,
        "upper_quantile": dataset["expected_return"] + 0.05,
        "confidence": np.full(dataset["expected_return"].shape, 0.5),
    }
    metrics = compute_supervised_metrics(prediction, dataset, groups)
    assert 0.0 <= metrics["ndcg_at_20"] <= 1.0
    assert -1.0 <= metrics["rank_ic"] <= 1.0
    assert metrics["return_mae"] >= 0.0


def test_stress_features_shape_and_mask():
    dataset = _build_dataset()
    gate = Phase5GateSpecV1()
    stress = build_stress_features(dataset["features"], dataset["valid_mask"], gate, seed=1)
    assert set(stress) == {"price", "volume", "missing", "extreme_volatility"}
    assert stress["price"]["features"].shape == dataset["features"].shape
    assert not np.allclose(stress["price"]["features"][:, :, 0], dataset["features"][:, :, 0])
    assert np.allclose(stress["price"]["features"][:, :, 19], dataset["features"][:, :, 19])
    assert (stress["missing"]["valid_mask"].astype(bool) <= dataset["valid_mask"].astype(bool)).all()


def test_train_fold_from_scratch_and_contract(tmp_path: Path):
    dataset = _build_dataset()
    spec = _specs()
    gate = Phase5GateSpecV1()
    artifact = train_phase5_supervised_fold(
        group_id="from_scratch", fold_id=1, dataset=dataset, split=FOLD,
        supervised_spec=spec, gate_spec=gate, gate_spec_sha256=gate.spec_sha256,
        init_checkpoint_path=None, output_dir=tmp_path / "out", supervised_budget=1,
    )
    normalized = validate_infots_fold_artifact(artifact, _contract(gate.spec_sha256))
    assert normalized["group_id"] == "from_scratch"
    assert normalized["test_blind"] is True
    assert (tmp_path / "out" / "from_scratch-fold-1-predictions.npz").is_file()
    assert len(artifact["embedding_snapshots"]) == 2
    assert artifact["selected_epoch"] >= 1


def test_train_fold_with_pretrained_init(tmp_path: Path):
    dataset = _build_dataset()
    checkpoint = _pretrain_checkpoint(dataset, tmp_path, FOLD["train_end"])
    state = load_encoder_state(checkpoint, 23, 16, 8)
    assert state and all(not key.startswith("encoder.") for key in state)
    spec = _specs()
    gate = Phase5GateSpecV1()
    artifact = train_phase5_supervised_fold(
        group_id="infots", fold_id=1, dataset=dataset, split=FOLD,
        supervised_spec=spec, gate_spec=gate, gate_spec_sha256=gate.spec_sha256,
        init_checkpoint_path=checkpoint, output_dir=tmp_path / "out2", supervised_budget=1,
    )
    assert artifact["init_checkpoint_sha256"] is not None
    validate_infots_fold_artifact(artifact, _contract(gate.spec_sha256))


def test_determinism_same_seed(tmp_path: Path):
    dataset = _build_dataset()
    spec = _specs()
    gate = Phase5GateSpecV1()
    first = train_phase5_supervised_fold(
        group_id="from_scratch", fold_id=1, dataset=dataset, split=FOLD,
        supervised_spec=spec, gate_spec=gate, gate_spec_sha256=gate.spec_sha256,
        init_checkpoint_path=None, output_dir=tmp_path / "a", supervised_budget=1,
    )
    second = train_phase5_supervised_fold(
        group_id="from_scratch", fold_id=1, dataset=dataset, split=FOLD,
        supervised_spec=spec, gate_spec=gate, gate_spec_sha256=gate.spec_sha256,
        init_checkpoint_path=None, output_dir=tmp_path / "b", supervised_budget=1,
    )
    assert first["metrics"] == second["metrics"]
    assert first["artifact_sha256"] == second["artifact_sha256"]


def test_fail_closed_guards(tmp_path: Path):
    dataset = _build_dataset()
    spec = _specs()
    gate = Phase5GateSpecV1()
    from python.qbt_ml.research.infots import InfoTSArtifactValidationError
    # from_scratch 带 checkpoint 失败
    checkpoint = _pretrain_checkpoint(dataset, tmp_path, FOLD["train_end"])
    with pytest.raises(InfoTSArtifactValidationError):
        train_phase5_supervised_fold(
            group_id="from_scratch", fold_id=1, dataset=dataset, split=FOLD,
            supervised_spec=spec, gate_spec=gate, gate_spec_sha256=gate.spec_sha256,
            init_checkpoint_path=checkpoint, output_dir=tmp_path / "x1", supervised_budget=1,
        )
    # 预训练组缺 checkpoint 失败
    with pytest.raises(InfoTSArtifactValidationError):
        train_phase5_supervised_fold(
            group_id="infots", fold_id=1, dataset=dataset, split=FOLD,
            supervised_spec=spec, gate_spec=gate, gate_spec_sha256=gate.spec_sha256,
            init_checkpoint_path=None, output_dir=tmp_path / "x2", supervised_budget=1,
        )
    # gate hash 不一致失败
    with pytest.raises(InfoTSArtifactValidationError):
        train_phase5_supervised_fold(
            group_id="from_scratch", fold_id=1, dataset=dataset, split=FOLD,
            supervised_spec=spec, gate_spec=gate, gate_spec_sha256="f" * 64,
            init_checkpoint_path=None, output_dir=tmp_path / "x3", supervised_budget=1,
        )
    # epochs 与 budget 不一致失败
    with pytest.raises(InfoTSArtifactValidationError):
        train_phase5_supervised_fold(
            group_id="from_scratch", fold_id=1, dataset=dataset, split=FOLD,
            supervised_spec=spec, gate_spec=gate, gate_spec_sha256=gate.spec_sha256,
            init_checkpoint_path=None, output_dir=tmp_path / "x4", supervised_budget=50,
        )
    # 破坏 purge gap 失败
    broken = dict(FOLD)
    broken["purge_gap"] = 10 * TIMESTAMP_STEP
    with pytest.raises(InfoTSArtifactValidationError):
        train_phase5_supervised_fold(
            group_id="from_scratch", fold_id=1, dataset=dataset, split=broken,
            supervised_spec=spec, gate_spec=gate, gate_spec_sha256=gate.spec_sha256,
            init_checkpoint_path=None, output_dir=tmp_path / "x5", supervised_budget=1,
        )


def test_regularized_training_runs(tmp_path: Path):
    dataset = _build_dataset()
    spec = _specs(dropout=0.1, weight_decay=1e-4, learning_rate=1e-4)
    gate = Phase5GateSpecV1()
    artifact = train_phase5_supervised_fold(
        group_id="from_scratch", fold_id=1, dataset=dataset, split=FOLD,
        supervised_spec=spec, gate_spec=gate, gate_spec_sha256=gate.spec_sha256,
        init_checkpoint_path=None, output_dir=tmp_path / "reg", supervised_budget=1,
    )
    validate_infots_fold_artifact(artifact, _contract(gate.spec_sha256))
    assert artifact["supervised_spec_sha256"] == spec.spec_sha256
