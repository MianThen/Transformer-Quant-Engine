from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


OUTPUT_NAMES = (
    "expected_return", "expected_volatility", "direction_probability",
    "lower_quantile", "upper_quantile", "confidence",
)


@dataclass(frozen=True)
class ModelManifest:
    schema_version: int
    model_id: str
    model_version: str
    model_sha256: str
    feature_profile: str
    feature_schema_sha256: str
    calendar_id: str
    universe_id: str
    data_cutoff_utc: str
    lookback: int
    feature_count: int
    static_feature_count: int
    outputs: tuple[dict, ...]
    frequency: str = "1d"
    execution_alignment: str = "NEXT_OPEN"
    input_dtype: str = "float32"
    input_layout: str = "NTF"
    onnx_opset: int = 17
    minimum_runtime_version: str = "1.17"
    preferred_provider: str = "CPUExecutionProvider"
    model_family: str | None = None
    architecture_version: str | None = None
    label_spec_sha256: str | None = None
    normalization_method: str | None = None
    normalization_sha256: str | None = None
    calibration_method: str | None = None
    calibration_sha256: str | None = None
    training_dataset_fingerprint: str | None = None
    leakage_report_sha256: str | None = None
    minimum_valid_tokens: int | None = None
    dynamic_batch: bool | None = None

    def validate(self) -> None:
        if self.schema_version not in {1, 2} or not self.model_id or not self.model_version:
            raise ValueError("manifest 版本或模型身份无效")
        if not self.calendar_id or not self.universe_id or not self.data_cutoff_utc:
            raise ValueError("manifest calendar、universe 或 data cutoff 无效")
        if self.feature_profile not in {"BAR_V1", "L1_QUOTE_V1", "TRADE_BAR_V1"}:
            raise ValueError("manifest feature_profile 无效")
        if self.lookback <= 0 or self.feature_count <= 0 or self.static_feature_count < 0:
            raise ValueError("manifest 输入 shape 无效")
        minute_frequency = (
            self.frequency.endswith("m") and self.frequency[:-1].isdigit()
            and int(self.frequency[:-1]) > 0
            and not self.frequency[:-1].startswith("0")
        )
        if self.frequency != "1d" and not minute_frequency:
            raise ValueError("manifest frequency 必须为 1d 或正整数分钟")
        if self.execution_alignment != "NEXT_OPEN":
            raise ValueError("第一版制品必须使用 NEXT_OPEN 对齐")
        names = tuple(value.get("name") for value in self.outputs)
        if names != OUTPUT_NAMES:
            raise ValueError("manifest 输出名称或顺序不符合预测协议")
        if self.schema_version == 2:
            expected_units = {
                "expected_return": "log_return",
                "expected_volatility": "return_std",
                "direction_probability": "probability",
                "lower_quantile": "log_return",
                "upper_quantile": "log_return",
                "confidence": "probability",
            }
            horizons = {value.get("horizon_bars") for value in self.outputs}
            if len(horizons) != 1 or next(iter(horizons), 0) <= 0:
                raise ValueError("Manifest V2 输出 horizon 必须相同且为正数")
            for value in self.outputs:
                if value.get("unit") != expected_units[value["name"]]:
                    raise ValueError("Manifest V2 输出单位无效")
            quantiles = {
                value["name"]: value.get("quantile") for value in self.outputs
                if value["name"] in {"lower_quantile", "upper_quantile"}
            }
            if quantiles != {"lower_quantile": 0.10, "upper_quantile": 0.90}:
                raise ValueError("Manifest V2 q10/q90 语义无效")
        hashes = [self.model_sha256, self.feature_schema_sha256]
        if self.schema_version == 2:
            required_text = (
                self.model_family, self.architecture_version, self.label_spec_sha256,
                self.normalization_method, self.normalization_sha256,
                self.calibration_method, self.calibration_sha256,
                self.training_dataset_fingerprint, self.leakage_report_sha256,
            )
            if any(not value for value in required_text):
                raise ValueError("Manifest V2 缺少训练、标签、归一化或校准 lineage")
            if self.minimum_valid_tokens is None or self.minimum_valid_tokens <= 0:
                raise ValueError("Manifest V2 minimum_valid_tokens 必须为正数")
            if self.dynamic_batch is not True:
                raise ValueError("Manifest V2 当前必须声明 dynamic_batch=true")
            if self.normalization_method != "mean_std":
                raise ValueError("Manifest V2 当前只支持 mean_std 归一化")
            if self.calibration_method != "platt_validation_only":
                raise ValueError("Manifest V2 当前只支持 validation-only Platt 校准")
            hashes.extend((
                self.label_spec_sha256, self.normalization_sha256,
                self.calibration_sha256, self.training_dataset_fingerprint,
                self.leakage_report_sha256,
            ))
        for value in hashes:
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("manifest SHA-256 格式无效")

    def write(self, path: str | Path) -> None:
        self.validate()
        value = {key: item for key, item in asdict(self).items() if item is not None}
        value["outputs"] = list(self.outputs)
        Path(path).write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> ModelManifest:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    value["outputs"] = tuple(value["outputs"])
    manifest = ModelManifest(**value)
    manifest.validate()
    return manifest


def validate_artifact(root: str | Path) -> ModelManifest:
    root = Path(root)
    required = ["model.onnx", "manifest.json", "feature_schema.json", "metrics.json"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError("模型制品缺少文件: " + ", ".join(missing))
    manifest = load_manifest(root / "manifest.json")
    if manifest.schema_version == 2:
        required.extend((
            "label_spec.json", "normalization.json", "calibration.json",
            "leakage_report.json",
        ))
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise ValueError("模型制品缺少文件: " + ", ".join(missing))
    if sha256_file(root / "model.onnx") != manifest.model_sha256:
        raise ValueError("model.onnx SHA-256 不匹配")
    if sha256_file(root / "feature_schema.json") != manifest.feature_schema_sha256:
        raise ValueError("feature_schema.json SHA-256 不匹配")
    if manifest.schema_version == 2:
        for name, expected in (
            ("label_spec.json", manifest.label_spec_sha256),
            ("normalization.json", manifest.normalization_sha256),
            ("calibration.json", manifest.calibration_sha256),
            ("leakage_report.json", manifest.leakage_report_sha256),
        ):
            if sha256_file(root / name) != expected:
                raise ValueError(f"{name} SHA-256 不匹配")
        label_spec = json.loads((root / "label_spec.json").read_text(encoding="utf-8"))
        normalization = json.loads(
            (root / "normalization.json").read_text(encoding="utf-8")
        )
        calibration = json.loads(
            (root / "calibration.json").read_text(encoding="utf-8")
        )
        leakage = json.loads(
            (root / "leakage_report.json").read_text(encoding="utf-8")
        )
        if normalization.get("method") != manifest.normalization_method:
            raise ValueError("normalization.json method 与 manifest 不一致")
        if calibration.get("method") != manifest.calibration_method:
            raise ValueError("calibration.json method 与 manifest 不一致")
        output_horizons = {value["horizon_bars"] for value in manifest.outputs}
        if output_horizons != {label_spec.get("horizon_bars")}:
            raise ValueError("label_spec.json horizon 与 manifest 输出不一致")
        if leakage.get("status") != "PASS":
            raise ValueError("Manifest V2 泄漏报告不是 PASS")
        if leakage.get("dataset_fingerprint") != manifest.training_dataset_fingerprint:
            raise ValueError("泄漏报告与训练数据 fingerprint 不一致")
    return manifest
