"""Phase 5 supervised-budget ablation runner: 2 pretrainings + 9 supervised folds + report.

预注册三组对照（from_scratch / fixed_augmentation / infots）× 3 folds。预训练语料固定为
fold-1 train_end 之前，绝不超过任何 fold 的 validation/test 边界。任何违规失败关闭。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from python.qbt_ml.evaluation.infots_ablation import (
    InfoTSBudgetContractV1,
    build_infots_ablation_report,
    validate_infots_fold_artifact,
)
from python.qbt_ml.research.infots import InfoTSAugmentationSpecV1
from python.qbt_ml.research.infots_supervised import (
    Phase5GateSpecV1,
    Phase5SupervisedSpecV1,
    train_phase5_supervised_fold,
)
from python.qbt_ml.research.infots_training import train_infots_pretraining

GROUP_IDS = ("from_scratch", "fixed_augmentation", "infots")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Phase 5 supervised config schema_version 必须为 1")
    if payload.get("production_eval") is not False:
        raise ValueError("Phase 5 supervised 禁止 production_eval")
    for key in ("dataset_sha256", "folds", "supervised", "gate_spec", "pretraining", "hypothesis_id"):
        if key not in payload:
            raise ValueError(f"config 缺少字段: {key}")
    if len(payload["folds"]) < 3:
        raise ValueError("必须提供至少三个 fold")
    return payload


def _subsample_rows(timestamps: np.ndarray, per_fold_limit: int) -> np.ndarray:
    """按 fold 时间序等距抽样，保留完整截面结构（每 fold 最多 per_fold_limit 个 timestamp）。"""
    keep: list[np.ndarray] = []
    for fold_first, fold_last in _fold_windows(timestamps):
        window = np.flatnonzero((timestamps >= fold_first) & (timestamps <= fold_last))
        if window.size == 0:
            continue
        unique = np.unique(timestamps[window])
        stride = max(1, unique.size // max(1, per_fold_limit))
        chosen = set(unique[::stride].tolist())
        keep.append(window[np.isin(timestamps[window], list(chosen))])
    return np.concatenate(keep) if keep else np.arange(timestamps.size)


_FOLD_BOUNDARIES: list[tuple[int, int]] = []


def _fold_windows(timestamps: np.ndarray) -> list[tuple[int, int]]:
    return _FOLD_BOUNDARIES or [(int(timestamps.min()), int(timestamps.max()))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--smoke", action="store_true", help="本地验证模式：CPU、1 epoch、按时间序抽样")
    parser.add_argument("--resume", action="store_true", help="fold artifact 已存在时跳过重训并复用（Kaggle 断点续跑）")
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    dataset_path = Path(args.dataset)
    dataset_sha256 = _sha256_file(dataset_path)
    if dataset_sha256 != config["dataset_sha256"]:
        raise ValueError(f"dataset SHA-256 mismatch: {dataset_sha256}")

    folds = config["folds"]
    global _FOLD_BOUNDARIES
    _FOLD_BOUNDARIES = [
        (int(fold["train_first"]), int(fold["test_end"])) for fold in folds
    ]

    supervised_settings = dict(config["supervised"])
    gate_settings = dict(config["gate_spec"])
    pretraining = config["pretraining"]
    if args.smoke:
        supervised_settings["epochs"] = 1
        supervised_settings["batch_size"] = 64
        supervised_settings["device"] = "cpu"
        pretraining = {**pretraining, "epochs": 1, "batch_size": 64, "device": "cpu"}

    supervised_spec = Phase5SupervisedSpecV1(**supervised_settings)
    gate_spec = Phase5GateSpecV1(**gate_settings)
    if gate_spec.spec_sha256 != config["gate_spec_sha256"]:
        raise ValueError("gate_spec_sha256 与 gate_spec 内容不一致")
    contract_payload = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    if args.smoke:
        contract_payload = {**contract_payload, "supervised_budget": 1}
    contract = InfoTSBudgetContractV1(**contract_payload)
    if contract.gate_spec_sha256 != gate_spec.spec_sha256:
        raise ValueError("预算合同与监督配置的 gate_spec_sha256 不一致")
    if supervised_spec.epochs != contract.supervised_budget and not args.smoke:
        raise ValueError("supervised epochs 必须等于合同 supervised_budget")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(dataset_path, allow_pickle=False) as loaded:
        required = {"features", "valid_mask", "timestamps", "symbols",
                    "expected_return", "direction", "realized_volatility",
                    "rank_relevance", "rank_utility"}
        missing = sorted(required - set(loaded.files))
        if missing:
            raise ValueError(f"dataset 缺少字段: {missing}")
        dataset = {key: loaded[key] for key in required}

    timestamps = np.asarray(dataset["timestamps"], dtype=np.int64)
    pretraining_corpus_end = int(folds[0]["train_end"])
    corpus_rows = np.flatnonzero(timestamps <= pretraining_corpus_end)
    if corpus_rows.size == 0:
        raise ValueError("预训练语料为空")
    if args.smoke:
        corpus_rows = _subsample_rows(timestamps[corpus_rows], per_fold_limit=4)

    checkpoint_paths: dict[str, Path] = {}
    pretraining_artifacts: dict[str, Any] = {}
    for group_id, augmentation_key in (("fixed_augmentation", "fixed_augmentation"), ("infots", "infots_pipeline")):
        augmentation_spec = InfoTSAugmentationSpecV1(**pretraining[augmentation_key])
        training_settings = dict(pretraining["training"])
        if args.smoke:
            training_settings = {**training_settings, "epochs": 1, "batch_size": 64, "device": "cpu"}
        from python.qbt_ml.research.infots_training import InfoTSTrainingSpecV1
        training_spec = InfoTSTrainingSpecV1(**training_settings)
        if training_spec.train_fold_end != pretraining_corpus_end:
            raise ValueError(f"{group_id} 预训练 train_fold_end 必须等于 fold-1 train_end")
        checkpoint = output_dir / f"pretraining-{group_id}.pt"
        if args.resume and checkpoint.is_file():
            print(json.dumps({"resume": "pretraining", "group": group_id}, ensure_ascii=False), flush=True)
        else:
            artifact = train_infots_pretraining(
                dataset["features"][corpus_rows],
                dataset["valid_mask"][corpus_rows],
                dataset["timestamps"][corpus_rows],
                augmentation_spec,
                training_spec,
                checkpoint_path=checkpoint,
            )
            pretraining_artifacts[group_id] = artifact
        checkpoint_paths[group_id] = checkpoint

    fold_artifacts: list[dict[str, Any]] = []
    budget = contract.supervised_budget if not args.smoke else 1
    for fold in folds:
        fold_id = int(fold["fold"])
        for group_id in GROUP_IDS:
            fold_dir = output_dir / f"fold-{fold_id}" / group_id
            artifact_path = fold_dir / "artifact.json"
            if args.resume and artifact_path.is_file():
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                validate_infots_fold_artifact(artifact, contract)
                print(json.dumps({"resume": "fold", "group": group_id, "fold": fold_id}, ensure_ascii=False), flush=True)
            else:
                artifact = train_phase5_supervised_fold(
                    group_id=group_id,
                    fold_id=fold_id,
                    dataset=dataset,
                    split=fold,
                    supervised_spec=supervised_spec,
                    gate_spec=gate_spec,
                    gate_spec_sha256=gate_spec.spec_sha256,
                    init_checkpoint_path=checkpoint_paths.get(group_id),
                    output_dir=fold_dir,
                    supervised_budget=budget,
                )
                validate_infots_fold_artifact(artifact, contract)
                artifact_path.write_text(
                    json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8",
                )
            fold_artifacts.append(artifact)
            print(json.dumps({
                "group": group_id, "fold": fold_id,
                "return_mae": round(artifact["metrics"]["return_mae"], 6),
                "direction_brier": round(artifact["metrics"]["direction_brier"], 6),
                "ndcg_at_20": round(artifact["metrics"]["ndcg_at_20"], 6),
                "selected_epoch": artifact["selected_epoch"],
            }, ensure_ascii=False), flush=True)

    report = build_infots_ablation_report(fold_artifacts, contract, evidence_level="REFERENCE_ONLY")
    report_path = output_dir / "phase5_infots_ablation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )

    validation = {
        "schema_version": 1,
        "mode": "smoke" if args.smoke else "full",
        "status": "PASS",
        "artifact_count": len(fold_artifacts),
        "pretraining_artifact_sha256": {
            group: pretraining_artifacts[group]["artifact_sha256"] for group in pretraining_artifacts
        },
        "report_sha256": report["report_sha256"],
        "gate_spec_sha256": gate_spec.spec_sha256,
        "contract_sha256": contract.contract_sha256,
        "dataset_sha256": dataset_sha256,
        "phase_exit_eligible": False,
        "promotion_eligible": False,
    }
    (output_dir / "RESULT_VALIDATION.json").write_text(
        json.dumps(validation, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "output": str(output_dir), "report": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
