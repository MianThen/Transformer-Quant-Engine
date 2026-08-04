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
    execution_alignment: str = "NEXT_OPEN"
    input_dtype: str = "float32"
    input_layout: str = "NTF"
    onnx_opset: int = 18
    minimum_runtime_version: str = "1.17"
    preferred_provider: str = "CPUExecutionProvider"
    label_spec_version: str = "V1"
    label_spec_sha256: str = ""
    ranking_score_spec: dict | None = None
    ranking_score_spec_sha256: str = ""
    ranking_loss_variant: str = "none"
    ranking_cutoff: int = 0
    ranking_temperature: float = 1.0
    rank_weight: float = 0.0

    def validate(self) -> None:
        if self.schema_version != 1 or not self.model_id or not self.model_version:
            raise ValueError("manifest 版本或模型身份无效")
        if self.feature_profile not in {"BAR_V1", "L1_QUOTE_V1", "TRADE_BAR_V1"}:
            raise ValueError("manifest feature_profile 无效")
        if self.lookback <= 0 or self.feature_count <= 0 or self.static_feature_count < 0:
            raise ValueError("manifest 输入 shape 无效")
        if self.execution_alignment != "NEXT_OPEN":
            raise ValueError("第一版制品必须使用 NEXT_OPEN 对齐")
        names = tuple(value.get("name") for value in self.outputs)
        if names != OUTPUT_NAMES:
            raise ValueError("manifest 输出名称或顺序不符合预测协议")
        for value in (self.model_sha256, self.feature_schema_sha256):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("manifest SHA-256 格式无效")
        if self.label_spec_version not in {"V1", "V2"}:
            raise ValueError("manifest label_spec_version 无效")
        if self.ranking_loss_variant not in {"none", "legacy", "listmle", "lambda"}:
            raise ValueError("manifest ranking_loss_variant 无效")
        if self.label_spec_version == "V2":
            for value in (self.label_spec_sha256, self.ranking_score_spec_sha256):
                if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                    raise ValueError("manifest LabelSpec/RankingScoreSpec SHA-256 无效")
            if not isinstance(self.ranking_score_spec, dict):
                raise ValueError("manifest 缺少 RankingScoreSpec V1")
            if self.ranking_cutoff <= 0 or self.ranking_temperature <= 0:
                raise ValueError("manifest ranking cutoff/temperature 无效")
            if self.rank_weight < 0:
                raise ValueError("manifest rank_weight 不能为负")

    def write(self, path: str | Path) -> None:
        self.validate()
        value = asdict(self)
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
    required = ("model.onnx", "manifest.json", "feature_schema.json", "metrics.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError("模型制品缺少文件: " + ", ".join(missing))
    manifest = load_manifest(root / "manifest.json")
    if sha256_file(root / "model.onnx") != manifest.model_sha256:
        raise ValueError("model.onnx SHA-256 不匹配")
    if sha256_file(root / "feature_schema.json") != manifest.feature_schema_sha256:
        raise ValueError("feature_schema.json SHA-256 不匹配")
    return manifest
