#!/usr/bin/env python3
"""Run the pre-registered Phase 1F top-k stability research experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


PHASE1E_CUTOFF = 1768374000000000000
TOP_K = 20
TEMPERATURE = 0.20
LAMBDA_STABILITY = 0.01
PURGE_TIMESTAMPS = 6
VALIDATION_COUNT = 59
SEED = 20260803
EPOCHS = 3
BATCH_CROSS_SECTIONS = 8
TRAIN_TIMESTAMP_STRIDE = 4


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _groups(timestamps: np.ndarray, indices: np.ndarray):
    selected = timestamps[indices]
    return [
        (int(timestamp), indices[np.flatnonzero(selected == timestamp)])
        for timestamp in np.unique(selected)
    ]


def _target(torch, data: dict[str, np.ndarray], indices: np.ndarray,
            symbol_tie_breaker: np.ndarray):
    return {
        "expected_return": torch.from_numpy(data["expected_return"][indices]),
        "direction": torch.from_numpy(data["direction"][indices]),
        "realized_volatility": torch.from_numpy(data["realized_volatility"][indices]),
        "rank_utility": torch.from_numpy(data["rank_utility"][indices]),
        "rank_relevance": torch.from_numpy(data["rank_relevance"][indices]),
        "timestamp": torch.from_numpy(data["timestamps"][indices].astype(np.int64)),
        "symbol_tie_breaker": torch.from_numpy(symbol_tie_breaker[indices]),
    }


def _train_model(data: dict[str, np.ndarray], train_indices: np.ndarray,
                 *, stability_enabled: bool, epochs: int):
    import torch
    from python.qbt_ml.models.temporal_transformer import (
        TemporalTransformerConfig, TemporalTransformerV1,
    )
    from python.qbt_ml.research.topk_stability import (
        torch_temporal_topk_stability_loss,
    )
    from python.qbt_ml.training.train import multitask_loss, seed_everything

    seed_everything(SEED)
    torch.set_num_threads(1)
    model_config = TemporalTransformerConfig(
        feature_count=int(data["features"].shape[2]),
        lookback=int(data["features"].shape[1]),
        d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1,
    )
    model = TemporalTransformerV1(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    symbols = data["symbols"].astype(str)
    symbol_order = {symbol: index for index, symbol in enumerate(sorted(set(symbols)))}
    symbol_tie_breaker = np.asarray([symbol_order[symbol] for symbol in symbols], dtype=np.int64)
    train_groups = _groups(data["timestamps"], train_indices)
    history: list[float] = []
    stability_values: list[float] = []

    for epoch in range(epochs):
        model.train()
        previous_scores = None
        previous_symbols = None
        epoch_loss = 0.0
        epoch_stability = []
        for batch_start in range(0, len(train_groups), BATCH_CROSS_SECTIONS):
            batch_groups = train_groups[batch_start:batch_start + BATCH_CROSS_SECTIONS]
            batch_indices = np.concatenate([indices for _, indices in batch_groups])
            features = torch.from_numpy(data["features"][batch_indices])
            valid_mask = torch.from_numpy(data["valid_mask"][batch_indices])
            target = _target(torch, data, batch_indices, symbol_tie_breaker)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features, valid_mask)
            base_loss = multitask_loss(
                prediction, target,
                return_weight=1.0, direction_weight=0.25,
                volatility_weight=0.25, quantile_weight=0.25,
                rank_weight=0.1, ranking_variant="legacy",
                ranking_cutoff=TOP_K, rank_temperature=1.0,
                ranking_score_mode="raw_return", risk_floor=1e-4,
            )
            stability_terms = []
            offset = 0
            for _, indices in batch_groups:
                size = indices.size
                current_scores = prediction["expected_return"][offset:offset + size]
                current_symbols = symbols[indices]
                if stability_enabled and previous_scores is not None:
                    term = torch_temporal_topk_stability_loss(
                        current_scores, current_symbols,
                        previous_scores, previous_symbols, TOP_K, TEMPERATURE,
                    )
                    stability_terms.append(term)
                    epoch_stability.append(float(term.detach()))
                previous_scores = current_scores.detach().clone()
                previous_symbols = current_symbols.copy()
                offset += size
            stability_loss = (
                torch.stack(stability_terms).mean()
                if stability_terms else prediction["expected_return"].sum() * 0.0
            )
            loss = base_loss + LAMBDA_STABILITY * stability_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Phase 1F loss 非有限: timestamp={timestamp}")
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(batch_groups)
        history.append(epoch_loss / max(1, len(train_groups)))
        stability_values.extend(epoch_stability)
        print(json.dumps({
            "event": "phase1f_epoch_completed", "epoch": epoch + 1,
            "epochs": epochs, "stability_enabled": stability_enabled,
            "train_loss": history[-1],
            "stability_penalty": float(np.mean(epoch_stability)) if epoch_stability else 0.0,
        }, sort_keys=True), flush=True)
    return model, model_config, history, stability_values, symbol_tie_breaker


def _predict(model, data: dict[str, np.ndarray], indices: np.ndarray,
             symbol_tie_breaker: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    import torch
    from python.qbt_ml.research.topk_stability import temporal_topk_stability_penalty
    from python.qbt_ml.training.ranking import ranking_oos_metrics

    model.eval()
    symbols = data["symbols"].astype(str)
    parts: dict[str, list[np.ndarray]] = {
        "scores": [], "utility": [], "relevance": [], "timestamps": [], "symbols": [],
        "expected_return": [], "expected_volatility": [], "direction_probability": [],
        "lower_quantile": [], "upper_quantile": [], "confidence": [],
    }
    with torch.no_grad():
        for _, group_indices in _groups(data["timestamps"], indices):
            feature = torch.from_numpy(data["features"][group_indices])
            valid_mask = torch.from_numpy(data["valid_mask"][group_indices])
            prediction = model(feature, valid_mask)
            score = prediction["expected_return"].numpy()
            parts["scores"].append(score)
            parts["utility"].append(data["rank_utility"][group_indices])
            parts["relevance"].append(data["rank_relevance"][group_indices])
            parts["timestamps"].append(data["timestamps"][group_indices])
            parts["symbols"].append(symbols[group_indices])
            for name in parts:
                if name in {"scores", "utility", "relevance", "timestamps", "symbols"}:
                    continue
                parts[name].append(prediction[name].numpy())
    values = {name: np.concatenate(chunks) for name, chunks in parts.items()}
    metrics = ranking_oos_metrics(
        values["scores"], values["utility"], values["relevance"],
        values["timestamps"], values["symbols"], cutoff=TOP_K,
    )
    stability = []
    grouped = []
    for timestamp in np.unique(values["timestamps"]):
        selected = values["timestamps"] == timestamp
        grouped.append({
            str(symbol): float(score)
            for symbol, score in zip(values["symbols"][selected], values["scores"][selected])
        })
    for previous, current in zip(grouped, grouped[1:]):
        stability.append(temporal_topk_stability_penalty(current, previous, TOP_K, TEMPERATURE))
    payload_metrics = {
        "cross_sections": metrics.cross_sections,
        "ndcg_at_cutoff": metrics.ndcg_at_cutoff,
        "rank_ic": metrics.rank_ic,
        "precision_at_cutoff": metrics.precision_at_cutoff,
        "top_k_overlap": metrics.top_k_overlap,
        "top_k_turnover": metrics.top_k_turnover,
        "top_bottom_utility_spread": metrics.top_bottom_utility_spread,
        "stability_penalty": float(np.mean(stability)) if stability else 0.0,
        "stability_transition_count": len(stability),
    }
    return values, payload_metrics


def _write_predictions(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def run(output: Path, dataset_path: Path, epochs: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Phase 1F 输出目录不可覆盖: {output}")
    output.mkdir(parents=True)
    with np.load(dataset_path, allow_pickle=False) as source:
        data = {
            key: source[key].copy() for key in (
                "features", "valid_mask", "timestamps", "symbols", "expected_return",
                "direction", "realized_volatility", "rank_utility", "rank_relevance",
            )
        }
    unique = np.unique(data["timestamps"])
    new_times = unique[unique > PHASE1E_CUTOFF]
    if new_times.size < VALIDATION_COUNT + 20:
        raise ValueError("Phase 1F 新窗口不足 validation/untouched 最小长度")
    validation_times = new_times[:VALIDATION_COUNT]
    untouched_times = new_times[VALIDATION_COUNT:]
    validation_start = int(np.searchsorted(unique, validation_times[0]))
    train_end = validation_start - PURGE_TIMESTAMPS
    train_times = unique[:train_end][::TRAIN_TIMESTAMP_STRIDE]
    train_indices = np.flatnonzero(np.isin(data["timestamps"], train_times))
    validation_indices = np.flatnonzero(np.isin(data["timestamps"], validation_times))
    untouched_indices = np.flatnonzero(np.isin(data["timestamps"], untouched_times))
    contract = {
        "schema_version": 1,
        "role": "phase1f_topk_stability_preregistered_contract",
        "registered_before_training": True,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "dataset_sha256": _file_hash(dataset_path),
        "phase1e_cutoff_timestamp": PHASE1E_CUTOFF,
        "train_first": int(train_times[0]), "train_last": int(train_times[-1]),
        "validation_first": int(validation_times[0]), "validation_last": int(validation_times[-1]),
        "untouched_first": int(untouched_times[0]), "untouched_last": int(untouched_times[-1]),
        "train_timestamps": int(train_times.size),
        "validation_timestamps": int(validation_times.size),
        "untouched_timestamps": int(untouched_times.size),
        "purge_timestamps": PURGE_TIMESTAMPS,
        "top_k": TOP_K,
        "temperature": TEMPERATURE,
        "lambda_stability": LAMBDA_STABILITY,
        "batch_cross_sections": BATCH_CROSS_SECTIONS,
        "train_timestamp_stride": TRAIN_TIMESTAMP_STRIDE,
        "epochs": int(epochs),
        "seed": SEED,
        "selection_policy": {
            "minimum_ndcg_delta": 0.0,
            "minimum_utility_spread_ratio": 1.0,
            "maximum_turnover_increase": 0.0,
            "maximum_stability_penalty_increase": 0.0,
        },
        "untouched_policy": "evaluate_one_frozen_winner_once",
        "future_label_policy": "training_and_validation_labels_are_filtered_by_timestamp; untouched_is_never_used_for_selection",
    }
    (output / "preregistered_contract.json").write_text(
        json.dumps({**contract, "contract_sha256": _canonical_hash(contract)},
                   ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    import torch
    _seed_everything(SEED)
    baseline_model, model_config, baseline_history, baseline_stability, tie_breaker = _train_model(
        data, train_indices, stability_enabled=False, epochs=epochs)
    _seed_everything(SEED)
    candidate_model, _, candidate_history, candidate_stability, _ = _train_model(
        data, train_indices, stability_enabled=True, epochs=epochs)
    baseline_validation, baseline_metrics = _predict(baseline_model, data, validation_indices, tie_breaker)
    candidate_validation, candidate_metrics = _predict(candidate_model, data, validation_indices, tie_breaker)
    candidate_pass = (
        candidate_metrics["ndcg_at_cutoff"] - baseline_metrics["ndcg_at_cutoff"] >= 0.0
        and candidate_metrics["top_bottom_utility_spread"] >= baseline_metrics["top_bottom_utility_spread"]
        and candidate_metrics["top_k_turnover"] <= baseline_metrics["top_k_turnover"]
        and candidate_metrics["stability_penalty"] <= baseline_metrics["stability_penalty"]
    )
    winner = "stability" if candidate_pass else "legacy"
    winner_model = candidate_model if winner == "stability" else baseline_model
    winner_validation = candidate_validation if winner == "stability" else baseline_validation
    winner_validation_metrics = candidate_metrics if winner == "stability" else baseline_metrics
    untouched_values, untouched_metrics = _predict(winner_model, data, untouched_indices, tie_breaker)
    _write_predictions(output / "validation_legacy_predictions.npz", baseline_validation)
    _write_predictions(output / "validation_stability_predictions.npz", candidate_validation)
    _write_predictions(output / "untouched_winner_predictions.npz", untouched_values)
    torch.save({
        "model_config": model_config.__dict__,
        "model_state_dict": winner_model.state_dict(),
        "winner": winner,
        "contract_sha256": _canonical_hash(contract),
        "history": candidate_history if winner == "stability" else baseline_history,
    }, output / "winner_checkpoint.pt")
    report = {
        "schema_version": 1,
        "role": "phase1f_topk_stability_training_report",
        "status": "winner_frozen_before_untouched",
        "evidence_tier": "RESEARCH_PROXY",
        "economic_claim_scope": "RESEARCH_PROXY_ONLY",
        "phase_exit_eligible": False,
        "promotion_eligible": False,
        "contract_sha256": _canonical_hash(contract),
        "dataset_sha256": _file_hash(dataset_path),
        "validation": {"legacy": baseline_metrics, "stability": candidate_metrics},
        "selection": {"candidate_pass": candidate_pass, "winner": winner},
        "winner_validation": winner_validation_metrics,
        "untouched": untouched_metrics,
        "training": {
            "legacy_loss": baseline_history,
            "stability_loss": candidate_history,
            "legacy_penalty_diagnostics": baseline_stability,
            "stability_penalty_diagnostics": candidate_stability,
        },
        "untouched_policy": "one_frozen_winner_once",
        "cpp_economic_gate": {"status": "pending_proxy_replay", "promotion_eligible": False},
    }
    report["report_sha256"] = _canonical_hash(report)
    (output / "phase1f_training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(
        "/Users/Zhuanz/PycharmProjects/PythonProject/data/research/phase1e_pit_120_dataset.npz"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase1f-topk-stability-real"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs 必须为正数")
    import sys
    project_root = Path("/Users/Zhuanz/PycharmProjects/PythonProject")
    sys.path.insert(0, str(project_root))
    report = run(args.output.resolve(), args.dataset.resolve(), args.epochs)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "report_sha256": report["report_sha256"],
        "winner": report["selection"]["winner"],
        "candidate_pass": report["selection"]["candidate_pass"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
