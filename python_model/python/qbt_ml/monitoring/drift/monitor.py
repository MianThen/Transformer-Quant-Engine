from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


class ReferenceKind(str, Enum):
    TRAINING_STATIC = "training_static"
    ROLLING_RECENT = "rolling_recent"


class LayerStatus(str, Enum):
    OK = "OK"
    PENDING_LABELS = "PENDING_LABELS"
    INCOMPATIBLE = "INCOMPATIBLE"
    UNAVAILABLE = "UNAVAILABLE"
    HARD_FAILURE = "HARD_FAILURE"


class AlertState(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {item.name: _json_value(getattr(value, item.name))
                for item in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class WindowSpec:
    start: int
    end: int
    available_at: int
    reference_kind: ReferenceKind | None = None

    def __post_init__(self) -> None:
        if self.start <= 0 or self.end < self.start or self.available_at < self.end:
            raise ValueError("window 时间边界或 available_at 无效")


@dataclass(frozen=True)
class EmbeddingSpec:
    checkpoint_sha256: str
    encoder_family: str
    layer_id: str
    pooling_spec_sha256: str
    dimension: int
    mask_spec_sha256: str

    def __post_init__(self) -> None:
        digests = (self.checkpoint_sha256, self.pooling_spec_sha256,
                   self.mask_spec_sha256)
        if (self.dimension <= 0 or not self.encoder_family or not self.layer_id or
                any(len(value) != 64 for value in digests)):
            raise ValueError("embedding compatibility spec 无效")


@dataclass(frozen=True)
class DriftMonitorSpecV1:
    schema_version: int = 1
    report_version: str = "drift-artifact-v1"
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
    psi_bins: int = 10
    top_k: int = 20
    minimum_sessions: int = 3
    minimum_observations: int = 20
    fast_window_sessions: int = 5
    confirm_window_sessions: int = 20
    bootstrap_replicates: int = 499
    mean_block_length: float = 5.0
    bootstrap_seed: int = 20260801
    fdr_q: float = 0.05
    persistence_windows: int = 2
    hard_missing_rate: float = 0.5
    marginal_warning_psi: float | None = None
    joint_warning_mmd: float | None = None
    concept_warning_ic_drop: float | None = None
    mmd_bandwidth: float = 1.0
    maximum_joint_samples: int = 512
    classifier_test_fraction: float = 0.3
    classifier_ridge: float = 1e-6
    stale_session_tolerance: int = 0

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.report_version:
            raise ValueError("DriftMonitorSpec V1 版本无效")
        if (self.psi_bins < 2 or self.top_k < 1 or self.minimum_sessions < 2 or
                self.minimum_observations < 2 or self.bootstrap_replicates < 1 or
                self.fast_window_sessions < self.minimum_sessions or
                self.confirm_window_sessions < self.fast_window_sessions or
                self.mean_block_length < 1.0 or self.bootstrap_seed == 0 or
                not 0.0 < self.fdr_q < 1.0 or self.persistence_windows < 1 or
                not 0.0 <= self.hard_missing_rate <= 1.0 or
                self.mmd_bandwidth <= 0.0 or self.maximum_joint_samples < 2):
            raise ValueError("DriftMonitorSpec V1 数值参数无效")
        if (not 0.0 < self.classifier_test_fraction < 0.5 or
                self.classifier_ridge <= 0.0 or self.stale_session_tolerance < 0):
            raise ValueError("two-sample/stale spec 无效")
        if (not self.quantiles or tuple(sorted(set(self.quantiles))) != self.quantiles or
                any(not 0.0 < value < 1.0 for value in self.quantiles)):
            raise ValueError("quantiles 必须严格递增且位于 (0, 1)")
        for value in (self.marginal_warning_psi, self.joint_warning_mmd,
                      self.concept_warning_ic_drop):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError("warning threshold 必须事前冻结为非负有限值")

    @property
    def sha256(self) -> str:
        return _sha256(_json_value(self))


def _eligible_sessions(frame: pd.DataFrame, *, session_column: str,
                       available_at: int, before_session: int | None = None) -> np.ndarray:
    if session_column not in frame.columns:
        raise ValueError(f"缺少 session column: {session_column}")
    eligible = frame
    if "available_at" in frame.columns:
        availability = pd.to_numeric(frame["available_at"], errors="coerce")
        if availability.isna().any():
            raise ValueError("available_at 必须完整且可解析")
        eligible = frame[availability <= available_at]
    if before_session is not None:
        eligible = eligible[eligible[session_column] < before_session]
    sessions = np.sort(eligible[session_column].dropna().unique())
    if sessions.size == 0:
        raise ValueError("没有满足 available_at 的 session")
    return sessions.astype(np.int64)


def build_reference_window(frame: pd.DataFrame, *, reference_kind: ReferenceKind,
                           session_column: str, available_at: int,
                           before_session: int,
                           training_start: int | None = None,
                           training_end: int | None = None,
                           rolling_sessions: int | None = None) -> WindowSpec:
    sessions = _eligible_sessions(
        frame, session_column=session_column, available_at=available_at,
        before_session=before_session)
    if reference_kind is ReferenceKind.TRAINING_STATIC:
        if training_start is None or training_end is None:
            raise ValueError("training_static 必须冻结 training_start/training_end")
        selected = sessions[(sessions >= training_start) & (sessions <= training_end)]
        if (selected.size == 0 or selected[0] != training_start or
                selected[-1] != training_end):
            raise ValueError("training_static session 与冻结边界不一致")
    else:
        if rolling_sessions is None or rolling_sessions < 2:
            raise ValueError("rolling_recent 必须冻结 rolling_sessions")
        if sessions.size < rolling_sessions:
            raise ValueError("rolling_recent 可用 session 不足")
        selected = sessions[-rolling_sessions:]
    return WindowSpec(int(selected[0]), int(selected[-1]), available_at,
                      reference_kind)


def build_current_windows(frame: pd.DataFrame, *, session_column: str,
                          available_at: int,
                          spec: DriftMonitorSpecV1) -> Mapping[str, WindowSpec]:
    sessions = _eligible_sessions(
        frame, session_column=session_column, available_at=available_at)
    if sessions.size < spec.confirm_window_sessions:
        raise ValueError("confirm window 可用 session 不足")
    fast = sessions[-spec.fast_window_sessions:]
    confirm = sessions[-spec.confirm_window_sessions:]
    return {
        "fast": WindowSpec(int(fast[0]), int(fast[-1]), available_at),
        "confirm": WindowSpec(int(confirm[0]), int(confirm[-1]), available_at),
    }


@dataclass(frozen=True)
class BootstrapResult:
    observed_mean: float
    p_value: float
    replicates: int
    mean_block_length: float
    seed: int
    replay_sha256: str


def stationary_bootstrap_mean(values: Sequence[float], *, replicates: int,
                              mean_block_length: float,
                              seed: int) -> BootstrapResult:
    data = np.asarray(values, dtype=np.float64)
    if (data.ndim != 1 or data.size < 2 or not np.isfinite(data).all()):
        raise ValueError("stationary bootstrap 需要至少两个有限观测")
    if replicates < 1 or mean_block_length < 1.0 or seed == 0:
        raise ValueError("stationary bootstrap spec 无效")
    observed = float(data.mean())
    centered = data - observed
    random = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    probability = 1.0 / mean_block_length
    for replicate in range(replicates):
        index = int(random.integers(0, data.size))
        total = 0.0
        for draw in range(data.size):
            total += centered[index]
            if draw + 1 < data.size:
                if random.random() < probability:
                    index = int(random.integers(0, data.size))
                else:
                    index = (index + 1) % data.size
        samples[replicate] = total / data.size
    exceedances = int(np.count_nonzero(np.abs(samples) >= abs(observed)))
    p_value = (exceedances + 1.0) / (replicates + 1.0)
    replay = _sha256({
        "observed": observed,
        "samples": samples.tolist(),
        "replicates": replicates,
        "mean_block_length": mean_block_length,
        "seed": seed,
    })
    return BootstrapResult(observed, p_value, replicates,
                           mean_block_length, seed, replay)


def _adjusted_p_values(p_values: Sequence[float], multiplier: float) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all() or (
            (values < 0.0) | (values > 1.0)).any():
        raise ValueError("p-values 必须是非空的 [0, 1] 有限向量")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 1.0
    for offset in range(values.size - 1, -1, -1):
        rank = offset + 1
        running = min(running, values[order[offset]] * values.size * multiplier / rank)
        adjusted[order[offset]] = min(1.0, running)
    return adjusted


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    return _adjusted_p_values(p_values, 1.0)


def benjamini_yekutieli(p_values: Sequence[float]) -> np.ndarray:
    count = len(p_values)
    return _adjusted_p_values(p_values, sum(1.0 / rank for rank in range(1, count + 1)))


def _finite_values(values: Sequence[float]) -> tuple[np.ndarray, int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    return finite, int(array.size - finite.size)


def _ks_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return float("nan")
    support = np.sort(np.unique(np.concatenate((left, right))))
    left_cdf = np.searchsorted(np.sort(left), support, side="right") / left.size
    right_cdf = np.searchsorted(np.sort(right), support, side="right") / right.size
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _psi(reference: np.ndarray, current: np.ndarray, missing_reference: int,
         missing_current: int, bins: int) -> tuple[float, tuple[float, ...]]:
    if reference.size == 0:
        return float("nan"), ()
    minimum, maximum = float(reference.min()), float(reference.max())
    interior = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    reference_counts = np.zeros(interior.size + 4, dtype=np.float64)
    current_counts = np.zeros_like(reference_counts)
    reference_counts[0] = missing_reference
    current_counts[0] = missing_current
    reference_counts[1] = np.count_nonzero(reference < minimum)
    current_counts[1] = np.count_nonzero(current < minimum)
    reference_counts[-1] = np.count_nonzero(reference > maximum)
    current_counts[-1] = np.count_nonzero(current > maximum)
    reference_inside = reference[(reference >= minimum) & (reference <= maximum)]
    current_inside = current[(current >= minimum) & (current <= maximum)]
    reference_counts[2:-1] = np.bincount(
        np.searchsorted(interior, reference_inside, side="right"),
        minlength=interior.size + 1)
    current_counts[2:-1] = np.bincount(
        np.searchsorted(interior, current_inside, side="right"),
        minlength=interior.size + 1)
    epsilon = 0.5
    reference_frequency = (reference_counts + epsilon) / (
        reference_counts.sum() + epsilon * reference_counts.size)
    current_frequency = (current_counts + epsilon) / (
        current_counts.sum() + epsilon * current_counts.size)
    value = np.sum((current_frequency - reference_frequency) *
                   np.log(current_frequency / reference_frequency))
    return float(value), tuple(float(item) for item in interior)


def continuous_drift(reference: Sequence[float], current: Sequence[float],
                     spec: DriftMonitorSpecV1) -> dict[str, Any]:
    reference_values, reference_missing = _finite_values(reference)
    current_values, current_missing = _finite_values(current)
    if reference_values.size < 2 or current_values.size < 2:
        return {"status": LayerStatus.UNAVAILABLE.value}
    psi, edges = _psi(reference_values, current_values, reference_missing,
                      current_missing, spec.psi_bins)

    def summary(values: np.ndarray, missing: int) -> dict[str, Any]:
        median = float(np.median(values))
        return {
            "count": int(values.size),
            "missing": missing,
            "missing_rate": missing / (values.size + missing),
            "mean": float(values.mean()),
            "variance": float(values.var(ddof=1)),
            "median": median,
            "mad": float(np.median(np.abs(values - median))),
            "quantiles": {str(q): float(np.quantile(values, q)) for q in spec.quantiles},
        }

    return {
        "status": LayerStatus.OK.value,
        "reference": summary(reference_values, reference_missing),
        "current": summary(current_values, current_missing),
        "psi": psi,
        "psi_reference_edges": edges,
        "ks_d": _ks_distance(reference_values, current_values),
    }


def categorical_drift(reference: Sequence[Any], current: Sequence[Any]) -> dict[str, Any]:
    left = pd.Series(reference, dtype="object").where(pd.notna(reference), "__MISSING__")
    right = pd.Series(current, dtype="object").where(pd.notna(current), "__MISSING__")
    if left.empty or right.empty:
        return {"status": LayerStatus.UNAVAILABLE.value}
    left = left.astype(str)
    right = right.astype(str)
    categories = sorted(set(left) | set(right))
    left_counts = left.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=np.float64)
    right_counts = right.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=np.float64)
    epsilon = 0.5
    left_frequency = (left_counts + epsilon) / (left_counts.sum() + epsilon * len(categories))
    right_frequency = (right_counts + epsilon) / (right_counts.sum() + epsilon * len(categories))
    midpoint = (left_frequency + right_frequency) / 2.0
    js = 0.5 * np.sum(left_frequency * np.log(left_frequency / midpoint))
    js += 0.5 * np.sum(right_frequency * np.log(right_frequency / midpoint))
    psi = np.sum((right_frequency - left_frequency) *
                 np.log(right_frequency / left_frequency))
    reference_categories = set(left)
    new_count = int(right.isin(set(right) - reference_categories).sum())
    return {
        "status": LayerStatus.OK.value,
        "categories": categories,
        "reference_frequency": dict(zip(categories, left_frequency.tolist())),
        "current_frequency": dict(zip(categories, right_frequency.tolist())),
        "psi": float(psi),
        "js_divergence": float(js),
        "total_variation": float(0.5 * np.abs(right_frequency - left_frequency).sum()),
        "new_category_rate": new_count / len(right),
    }


def _ordered_rows(values: np.ndarray, limit: int) -> np.ndarray:
    order = np.lexsort(tuple(values[:, column] for column in range(values.shape[1] - 1, -1, -1)))
    ordered = values[order]
    if ordered.shape[0] <= limit:
        return ordered
    indices = np.linspace(0, ordered.shape[0] - 1, limit, dtype=np.int64)
    return ordered[indices]


def _effective_rank(covariance: np.ndarray) -> float:
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return 0.0
    probabilities = eigenvalues[eigenvalues > 0.0] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _mmd_rbf(left: np.ndarray, right: np.ndarray, bandwidth: float) -> float:
    def kernel(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        squared = np.sum((first[:, None, :] - second[None, :, :]) ** 2, axis=2)
        return np.exp(-squared / (2.0 * bandwidth * bandwidth))

    left_kernel = kernel(left, left)
    right_kernel = kernel(right, right)
    cross_kernel = kernel(left, right)
    left_term = ((left_kernel.sum() - np.trace(left_kernel)) /
                 (left.shape[0] * (left.shape[0] - 1)))
    right_term = ((right_kernel.sum() - np.trace(right_kernel)) /
                  (right.shape[0] * (right.shape[0] - 1)))
    return float(max(0.0, left_term + right_term - 2.0 * cross_kernel.mean()))


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=np.float64)
    value = ((ranks[labels == 1].sum() -
              positives * (positives + 1) / 2.0) /
             (positives * negatives))
    return float(max(value, 1.0 - value))


def _two_sample_auc(left: np.ndarray, right: np.ndarray,
                    spec: DriftMonitorSpecV1) -> float:
    if min(left.shape[0], right.shape[0]) < 4:
        return float("nan")
    random = np.random.default_rng(spec.bootstrap_seed)
    left_order = random.permutation(left.shape[0])
    right_order = random.permutation(right.shape[0])
    left_test_count = max(1, int(left.shape[0] * spec.classifier_test_fraction))
    right_test_count = max(1, int(right.shape[0] * spec.classifier_test_fraction))
    left_test, left_train = left[left_order[:left_test_count]], left[left_order[left_test_count:]]
    right_test, right_train = right[right_order[:right_test_count]], right[right_order[right_test_count:]]
    pooled = np.vstack((left_train, right_train))
    location = pooled.mean(axis=0)
    scale = pooled.std(axis=0, ddof=1)
    scale[scale == 0.0] = 1.0
    left_train = (left_train - location) / scale
    right_train = (right_train - location) / scale
    pooled_train = np.vstack((left_train, right_train))
    covariance = np.atleast_2d(np.cov(pooled_train, rowvar=False, ddof=1))
    covariance += np.eye(covariance.shape[0]) * spec.classifier_ridge
    direction = np.linalg.solve(
        covariance, right_train.mean(axis=0) - left_train.mean(axis=0))
    test = np.vstack(((left_test - location) / scale, (right_test - location) / scale))
    labels = np.concatenate((np.zeros(left_test.shape[0], dtype=np.int8),
                             np.ones(right_test.shape[0], dtype=np.int8)))
    return _auc(labels, test @ direction)


def linear_cka(left: Sequence[Sequence[float]],
               right: Sequence[Sequence[float]]) -> float:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if (first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[0] or
            first.shape[0] < 2 or not np.isfinite(first).all() or
            not np.isfinite(second).all()):
        raise ValueError("CKA 需要相同 anchor 行数的有限二维矩阵")
    first = first - first.mean(axis=0, keepdims=True)
    second = second - second.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(first.T @ second, ord="fro") ** 2
    first_norm = np.linalg.norm(first.T @ first, ord="fro")
    second_norm = np.linalg.norm(second.T @ second, ord="fro")
    denominator = first_norm * second_norm
    return float(cross / denominator) if denominator > 0.0 else 0.0


def joint_drift(reference: Sequence[Sequence[float]],
                current: Sequence[Sequence[float]],
                spec: DriftMonitorSpecV1) -> dict[str, Any]:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(current, dtype=np.float64)
    if (left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]):
        raise ValueError("joint drift 输入维度不兼容")
    left = left[np.isfinite(left).all(axis=1)]
    right = right[np.isfinite(right).all(axis=1)]
    if min(left.shape[0], right.shape[0]) < 2:
        return {"status": LayerStatus.UNAVAILABLE.value}
    left = _ordered_rows(left, spec.maximum_joint_samples)
    right = _ordered_rows(right, spec.maximum_joint_samples)
    left_cov = np.atleast_2d(np.cov(left, rowvar=False, ddof=1))
    right_cov = np.atleast_2d(np.cov(right, rowvar=False, ddof=1))
    left_corr = np.nan_to_num(np.corrcoef(left, rowvar=False), nan=0.0)
    right_corr = np.nan_to_num(np.corrcoef(right, rowvar=False), nan=0.0)
    if left_corr.ndim == 0:
        left_corr = right_corr = np.ones((1, 1), dtype=np.float64)
    denominator = max(float(np.linalg.norm(left_corr, ord="fro")), 1e-12)
    left_eigen = np.sort(np.maximum(np.linalg.eigvalsh(left_cov), 0.0))[::-1]
    right_eigen = np.sort(np.maximum(np.linalg.eigvalsh(right_cov), 0.0))[::-1]
    return {
        "status": LayerStatus.OK.value,
        "correlation_frobenius_distance": float(
            np.linalg.norm(right_corr - left_corr, ord="fro") / denominator),
        "eigenvalue_l2_distance": float(np.linalg.norm(right_eigen - left_eigen)),
        "reference_effective_rank": _effective_rank(left_cov),
        "current_effective_rank": _effective_rank(right_cov),
        "mmd_rbf": _mmd_rbf(left, right, spec.mmd_bandwidth),
        "classifier_two_sample_auc": _two_sample_auc(left, right, spec),
    }


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left, ddof=1) == 0.0 or np.std(right, ddof=1) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _ndcg(scores: np.ndarray, relevance: np.ndarray, cutoff: int) -> float:
    count = min(cutoff, scores.size)
    gains = np.exp2(np.maximum(relevance, 0.0)) - 1.0
    discounts = 1.0 / np.log2(np.arange(count, dtype=np.float64) + 2.0)
    actual = float(np.dot(gains[np.argsort(-scores, kind="stable")[:count]], discounts))
    ideal = float(np.dot(np.sort(gains)[::-1][:count], discounts))
    return actual / ideal if ideal > 0.0 else 0.0


def concept_metrics(frame: pd.DataFrame, *, session_column: str,
                    score_column: str, return_column: str,
                    utility_column: str | None, top_k: int) -> dict[str, Any]:
    required = {session_column, score_column, return_column}
    if not required.issubset(frame.columns):
        return {"status": LayerStatus.UNAVAILABLE.value}
    pearson, rank_ic, ndcg, spreads = [], [], [], []
    for _, group in frame.groupby(session_column, sort=True):
        valid = group[[score_column, return_column]].replace(
            [np.inf, -np.inf], np.nan).dropna()
        if valid.shape[0] < 2:
            continue
        scores = valid[score_column].to_numpy(dtype=np.float64)
        returns = valid[return_column].to_numpy(dtype=np.float64)
        pearson.append(_correlation(scores, returns))
        rank_ic.append(_correlation(_rank(scores), _rank(returns)))
        relevance = returns - min(float(returns.min()), 0.0)
        ndcg.append(_ndcg(scores, relevance, top_k))
        order = np.argsort(scores, kind="stable")
        count = min(top_k, max(1, scores.size // 2))
        spreads.append(float(returns[order[-count:]].mean() - returns[order[:count]].mean()))
    pearson_values = np.asarray(pearson, dtype=np.float64)
    rank_values = np.asarray(rank_ic, dtype=np.float64)
    pearson_values = pearson_values[np.isfinite(pearson_values)]
    rank_values = rank_values[np.isfinite(rank_values)]
    if pearson_values.size == 0 or rank_values.size == 0:
        return {"status": LayerStatus.UNAVAILABLE.value}
    ic_std = float(pearson_values.std(ddof=1)) if pearson_values.size > 1 else 0.0
    result = {
        "status": LayerStatus.OK.value,
        "pearson_ic": float(pearson_values.mean()),
        "rank_ic": float(rank_values.mean()),
        "icir": float(pearson_values.mean() / ic_std) if ic_std > 0.0 else 0.0,
        "ic_sign_rate": float(np.mean(pearson_values > 0.0)),
        "ndcg_at_k": float(np.mean(ndcg)),
        "utility_spread": float(np.mean(spreads)),
        "sessions": int(pearson_values.size),
    }
    merged = frame.replace([np.inf, -np.inf], np.nan)
    if "expected_return" in merged.columns:
        valid = merged[["expected_return", return_column]].dropna()
        result["return_mae"] = float(np.mean(np.abs(
            valid["expected_return"].to_numpy() - valid[return_column].to_numpy())))
    if "direction_probability" in merged.columns:
        valid = merged[["direction_probability", return_column]].dropna()
        probability = np.clip(valid["direction_probability"].to_numpy(), 0.0, 1.0)
        direction = (valid[return_column].to_numpy() > 0.0).astype(np.float64)
        result["direction_brier"] = float(np.mean((probability - direction) ** 2))
    if "q10" in merged.columns and "q90" in merged.columns:
        valid = merged[["q10", "q90", return_column]].dropna()
        result["interval_coverage"] = float(np.mean(
            (valid[return_column] >= valid["q10"]) &
            (valid[return_column] <= valid["q90"])))
        result["quantile_crossing_rate"] = float(np.mean(valid["q10"] > valid["q90"]))
    precision = []
    for _, group in merged.groupby(session_column, sort=True):
        valid = group[[score_column, return_column]].dropna()
        if valid.empty:
            continue
        selected = valid.nlargest(min(top_k, len(valid)), score_column)
        precision.append(float(np.mean(selected[return_column] > 0.0)))
    result["precision_at_k"] = float(np.mean(precision)) if precision else None
    if utility_column and utility_column in frame.columns:
        valid = frame[[score_column, utility_column]].replace(
            [np.inf, -np.inf], np.nan).dropna()
        result["utility_mae"] = float(np.mean(np.abs(
            valid[score_column].to_numpy() - valid[utility_column].to_numpy())))
    return result


def _frame_hash(frame: pd.DataFrame) -> str:
    ordered = frame.sort_index(axis=1).sort_values(
        list(frame.columns), kind="stable", na_position="first")
    values = pd.util.hash_pandas_object(ordered, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _array_hash(values: Sequence[Sequence[float]]) -> str:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("snapshot array 必须是二维")
    ordered = _ordered_rows(array, array.shape[0])
    return hashlib.sha256(ordered.tobytes()).hexdigest()


def _window(frame: pd.DataFrame, window: WindowSpec, session_column: str) -> pd.DataFrame:
    if session_column not in frame.columns:
        raise ValueError(f"缺少 session column: {session_column}")
    if "available_at" in frame.columns:
        available = pd.to_numeric(frame["available_at"], errors="coerce")
        if available.isna().any():
            raise ValueError("available_at 必须完整且可解析")
        leaked = frame[(frame[session_column] >= window.start) &
                       (frame[session_column] <= window.end) &
                       (available > window.available_at)]
        if not leaked.empty:
            raise ValueError("snapshot 包含 available_at 之后才可用的数据")
    selected = frame[(frame[session_column] >= window.start) &
                     (frame[session_column] <= window.end)].copy()
    if selected.empty:
        raise ValueError("window 没有数据")
    return selected


def _universe_summary(reference: pd.DataFrame, current: pd.DataFrame,
                      session_column: str, symbol_column: str) -> dict[str, Any]:
    reference_symbols = set(reference[symbol_column].astype(str))
    current_symbols = set(current[symbol_column].astype(str))
    union = reference_symbols | current_symbols
    overlap = len(reference_symbols & current_symbols) / len(union) if union else 1.0
    return {
        "reference_count": len(reference_symbols),
        "current_count": len(current_symbols),
        "composition_jaccard": overlap,
        "new_symbol_rate": len(current_symbols - reference_symbols) /
                           max(1, len(current_symbols)),
        "reference_session_counts": reference.groupby(
            session_column)[symbol_column].nunique().tolist(),
        "current_session_counts": current.groupby(
            session_column)[symbol_column].nunique().tolist(),
    }


def _prediction_portfolio_summary(frame: pd.DataFrame, session_column: str,
                                  symbol_column: str, score_column: str,
                                  top_k: int) -> dict[str, Any]:
    top_sets: list[set[str]] = []
    concentration = []
    for _, group in frame.groupby(session_column, sort=True):
        valid = group[[symbol_column, score_column]].replace(
            [np.inf, -np.inf], np.nan).dropna()
        if valid.empty:
            continue
        ordered = valid.sort_values([score_column, symbol_column], kind="stable")
        selected = ordered.tail(min(top_k, len(ordered)))
        top_sets.append(set(selected[symbol_column].astype(str)))
        scores = selected[score_column].to_numpy(dtype=np.float64)
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        concentration.append(float(np.sum(weights * weights)))
    overlap = []
    for previous, current in zip(top_sets, top_sets[1:]):
        overlap.append(len(previous & current) / max(1, len(previous | current)))
    return {
        "topk_overlap": float(np.mean(overlap)) if overlap else None,
        "topk_turnover": float(np.mean([1.0 - value for value in overlap])) if overlap else None,
        "topk_concentration": float(np.mean(concentration)) if concentration else None,
        "sessions": len(top_sets),
    }


@dataclass(frozen=True)
class DriftReport:
    schema_version: int
    report_version: str
    monitor_spec_sha256: str
    reference_window: WindowSpec
    current_window: WindowSpec
    data_quality: Mapping[str, Any]
    raw_drift: Mapping[str, Any]
    feature_drift: Mapping[str, Any]
    feature_joint_drift: Mapping[str, Any]
    prediction_drift: Mapping[str, Any]
    label_status: LayerStatus
    label_drift: Mapping[str, Any] | None
    concept_status: LayerStatus
    concept_performance: Mapping[str, Any] | None
    embedding_status: LayerStatus
    embedding_drift: Mapping[str, Any] | None
    alert_state: AlertState
    alert_reasons: tuple[str, ...]
    persistence_count: int
    retraining_review_recommended: bool
    artifact_hashes: Mapping[str, str]
    source_snapshot_set_sha256: str
    report_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)

    def verify_hash(self) -> bool:
        payload = self.to_dict()
        report_sha256 = payload.pop("report_sha256")
        return _sha256(payload) == report_sha256


@dataclass
class DriftAlertMachine:
    spec: DriftMonitorSpecV1
    persistence_count: int = 0
    last_state: AlertState = AlertState.INFO

    def update(self, *, hard_failure: bool, marginal_signal: bool,
               joint_signal: bool, concept_signal: bool,
               reasons: Iterable[str]) -> tuple[AlertState, int, tuple[str, ...]]:
        reason_tuple = tuple(sorted(set(reasons)))
        if hard_failure:
            self.persistence_count += 1
            self.last_state = AlertState.CRITICAL
            return self.last_state, self.persistence_count, reason_tuple
        distribution_signal = marginal_signal or joint_signal
        if distribution_signal or concept_signal:
            self.persistence_count += 1
        else:
            self.persistence_count = 0
        if (distribution_signal and concept_signal and
                self.persistence_count >= self.spec.persistence_windows):
            self.last_state = AlertState.CRITICAL
        elif distribution_signal or concept_signal:
            self.last_state = AlertState.WARN
        else:
            self.last_state = AlertState.INFO
        return self.last_state, self.persistence_count, reason_tuple


def build_drift_report(*, reference_raw: pd.DataFrame, current_raw: pd.DataFrame,
                       reference_features: pd.DataFrame, current_features: pd.DataFrame,
                       reference_predictions: pd.DataFrame,
                       current_predictions: pd.DataFrame,
                       reference_window: WindowSpec, current_window: WindowSpec,
                       raw_columns: Sequence[str], feature_columns: Sequence[str],
                       prediction_columns: Sequence[str],
                       categorical_raw_columns: Sequence[str] = (),
                       session_column: str = "timestamp",
                       symbol_column: str = "symbol", score_column: str = "ranking_score",
                       reference_labels: pd.DataFrame | None = None,
                       current_labels: pd.DataFrame | None = None,
                       return_column: str = "return_raw",
                       label_columns: Sequence[str] = (),
                       utility_column: str | None = "rank_utility",
                       reference_embeddings: Sequence[Sequence[float]] | None = None,
                       current_embeddings: Sequence[Sequence[float]] | None = None,
                       reference_embedding_spec: EmbeddingSpec | None = None,
                       current_embedding_spec: EmbeddingSpec | None = None,
                       reference_anchor_embeddings: Sequence[Sequence[float]] | None = None,
                       current_anchor_embeddings: Sequence[Sequence[float]] | None = None,
                       fixed_anchor_sha256: str | None = None,
                       source_snapshot_hashes: Sequence[str] = (),
                       artifact_hashes: Mapping[str, str] | None = None,
                       alert_machine: DriftAlertMachine | None = None,
                       spec: DriftMonitorSpecV1 = DriftMonitorSpecV1()) -> DriftReport:
    if reference_window.reference_kind is None:
        raise ValueError("reference window 必须声明 reference_kind")
    if reference_window.end >= current_window.start:
        raise ValueError("reference 与 current window 必须按时间隔离")
    if reference_window.available_at > current_window.available_at:
        raise ValueError("reference available_at 不得晚于 current")
    reference_raw = _window(reference_raw, reference_window, session_column)
    current_raw = _window(current_raw, current_window, session_column)
    reference_features = _window(reference_features, reference_window, session_column)
    current_features = _window(current_features, current_window, session_column)
    reference_predictions = _window(reference_predictions, reference_window, session_column)
    current_predictions = _window(current_predictions, current_window, session_column)

    required_base = {session_column, symbol_column}
    missing_columns = sorted(
        (required_base | set(raw_columns) | set(categorical_raw_columns)) -
        set(reference_raw.columns) |
        (required_base | set(raw_columns) | set(categorical_raw_columns)) -
        set(current_raw.columns) |
        (required_base | set(feature_columns)) - set(reference_features.columns) |
        (required_base | set(feature_columns)) - set(current_features.columns) |
        (required_base | set(prediction_columns) | {score_column}) -
        set(reference_predictions.columns) |
        (required_base | set(prediction_columns) | {score_column}) -
        set(current_predictions.columns))
    duplicate_count = int(current_raw.duplicated(
        [session_column, symbol_column]).sum())
    current_missing = current_raw[
        list(set(raw_columns) & set(current_raw.columns))].isna().mean()
    maximum_missing = float(current_missing.max()) if not current_missing.empty else 0.0
    last_session_by_symbol = current_raw.groupby(symbol_column)[session_column].max()
    stale_cutoff = current_window.end - spec.stale_session_tolerance
    stale_symbols = sorted(last_session_by_symbol[
        last_session_by_symbol < stale_cutoff].index.astype(str))
    stale_rate = len(stale_symbols) / max(1, last_session_by_symbol.size)
    hard_failure = bool(
        missing_columns or duplicate_count or
        maximum_missing >= spec.hard_missing_rate or
        stale_rate >= spec.hard_missing_rate)
    data_quality = {
        "status": LayerStatus.HARD_FAILURE.value if hard_failure else LayerStatus.OK.value,
        "missing_columns": missing_columns,
        "duplicate_session_symbol_rows": duplicate_count,
        "maximum_raw_missing_rate": maximum_missing,
        "stale_symbols": stale_symbols,
        "stale_symbol_rate": stale_rate,
        "universe": _universe_summary(reference_raw, current_raw,
                                      session_column, symbol_column),
    }

    raw_summary = {column: continuous_drift(reference_raw[column], current_raw[column], spec)
                   for column in raw_columns if column in reference_raw and column in current_raw}
    raw_summary.update({
        column: categorical_drift(reference_raw[column], current_raw[column])
        for column in categorical_raw_columns
        if column in reference_raw and column in current_raw
    })
    feature_summary = {
        column: continuous_drift(reference_features[column], current_features[column], spec)
        for column in feature_columns if column in reference_features and column in current_features
    }
    feature_joint = joint_drift(reference_features[list(feature_columns)].to_numpy(),
                                current_features[list(feature_columns)].to_numpy(), spec)
    prediction_summary = {
        column: continuous_drift(reference_predictions[column],
                                 current_predictions[column], spec)
        for column in tuple(prediction_columns) + (score_column,)
        if column in reference_predictions and column in current_predictions
    }
    prediction_summary["portfolio"] = {
        "reference": _prediction_portfolio_summary(
            reference_predictions, session_column, symbol_column, score_column, spec.top_k),
        "current": _prediction_portfolio_summary(
            current_predictions, session_column, symbol_column, score_column, spec.top_k),
    }

    labels_ready = reference_labels is not None and current_labels is not None
    label_status = LayerStatus.PENDING_LABELS
    label_summary = None
    concept_status = LayerStatus.PENDING_LABELS
    concept_reference = concept_current = None
    if labels_ready:
        reference_labels = _window(reference_labels, reference_window, session_column)
        current_labels = _window(current_labels, current_window, session_column)
        requested_labels = tuple(dict.fromkeys((return_column, *label_columns)))
        available_labels = tuple(
            column for column in requested_labels
            if column in reference_labels and column in current_labels)
        if available_labels:
            label_status = LayerStatus.OK
            label_summary = {
                column: continuous_drift(reference_labels[column],
                                         current_labels[column], spec)
                for column in available_labels
            }
        reference_merged = reference_predictions.merge(
            reference_labels, on=[session_column, symbol_column], how="inner",
            suffixes=("", "_label"))
        current_merged = current_predictions.merge(
            current_labels, on=[session_column, symbol_column], how="inner",
            suffixes=("", "_label"))
        concept_reference = concept_metrics(
            reference_merged, session_column=session_column, score_column=score_column,
            return_column=return_column, utility_column=utility_column, top_k=spec.top_k)
        concept_current = concept_metrics(
            current_merged, session_column=session_column, score_column=score_column,
            return_column=return_column, utility_column=utility_column, top_k=spec.top_k)
        if (concept_reference.get("status") == LayerStatus.OK.value and
                concept_current.get("status") == LayerStatus.OK.value):
            concept_status = LayerStatus.OK
    concept_summary = None if concept_reference is None else {
        "reference": concept_reference,
        "current": concept_current,
        "pearson_ic_change": concept_current.get("pearson_ic", 0.0) -
                             concept_reference.get("pearson_ic", 0.0),
        "rank_ic_change": concept_current.get("rank_ic", 0.0) -
                          concept_reference.get("rank_ic", 0.0),
    }

    embedding_status = LayerStatus.UNAVAILABLE
    embedding_summary = None
    if reference_embeddings is not None and current_embeddings is not None:
        if (reference_embedding_spec is None or current_embedding_spec is None or
                reference_embedding_spec != current_embedding_spec):
            embedding_status = LayerStatus.INCOMPATIBLE
        else:
            embedding_status = LayerStatus.OK
            embedding_summary = joint_drift(reference_embeddings, current_embeddings, spec)
            embedding_summary["diagnostic_only"] = True
            if (reference_anchor_embeddings is not None and
                    current_anchor_embeddings is not None):
                if fixed_anchor_sha256 is None or len(fixed_anchor_sha256) != 64:
                    raise ValueError("CKA 必须冻结 fixed_anchor_sha256")
                embedding_summary["fixed_anchor_sha256"] = fixed_anchor_sha256
                embedding_summary["linear_cka"] = linear_cka(
                    reference_anchor_embeddings, current_anchor_embeddings)

    p_values, p_columns = [], []
    for column, summary in feature_summary.items():
        if summary.get("status") != LayerStatus.OK.value:
            continue
        current_daily = current_features.groupby(session_column)[column].mean().dropna()
        if current_daily.size >= spec.minimum_sessions:
            differences = current_daily.to_numpy() - summary["reference"]["mean"]
            bootstrap = stationary_bootstrap_mean(
                differences, replicates=spec.bootstrap_replicates,
                mean_block_length=spec.mean_block_length, seed=spec.bootstrap_seed)
            summary["stationary_bootstrap"] = _json_value(bootstrap)
            p_values.append(bootstrap.p_value)
            p_columns.append(column)
    if p_values:
        bh = benjamini_hochberg(p_values)
        by = benjamini_yekutieli(p_values)
        for index, column in enumerate(p_columns):
            feature_summary[column]["bh_q"] = float(bh[index])
            feature_summary[column]["by_q"] = float(by[index])

    reasons = []
    if hard_failure:
        reasons.append("DATA_QUALITY_HARD_FAILURE")
    marginal_signal = False
    if spec.marginal_warning_psi is not None:
        marginal_signal = any(
            item.get("psi", 0.0) >= spec.marginal_warning_psi
            for item in feature_summary.values())
        if marginal_signal:
            reasons.append("FEATURE_MARGINAL_DRIFT")
    joint_signal = (spec.joint_warning_mmd is not None and
                    feature_joint.get("mmd_rbf", 0.0) >= spec.joint_warning_mmd)
    if joint_signal:
        reasons.append("FEATURE_JOINT_DRIFT")
    concept_signal = False
    if (concept_status is LayerStatus.OK and spec.concept_warning_ic_drop is not None and
            concept_summary is not None):
        concept_signal = concept_summary["pearson_ic_change"] <= -spec.concept_warning_ic_drop
        if concept_signal:
            reasons.append("CONCEPT_IC_DEGRADATION")
    machine = alert_machine or DriftAlertMachine(spec)
    state, persistence, reason_tuple = machine.update(
        hard_failure=hard_failure, marginal_signal=marginal_signal,
        joint_signal=joint_signal, concept_signal=concept_signal, reasons=reasons)

    normalized_hashes = dict(artifact_hashes or {
        f"source_{index}": value
        for index, value in enumerate(source_snapshot_hashes)
    })
    if any(len(value) != 64 or any(character not in "0123456789abcdef"
                                   for character in value)
           for value in normalized_hashes.values()):
        raise ValueError("artifact hashes 必须是小写 SHA-256")
    source_frames = [
        reference_raw, current_raw, reference_features, current_features,
        reference_predictions, current_predictions,
    ]
    if labels_ready:
        source_frames.extend((reference_labels, current_labels))
    source_payload = {
        "artifact_hashes": normalized_hashes,
        "reference_window": _json_value(reference_window),
        "current_window": _json_value(current_window),
        "frames": [_frame_hash(frame) for frame in source_frames],
        "embeddings": [] if (reference_embeddings is None or
                              current_embeddings is None) else [
            _array_hash(reference_embeddings), _array_hash(current_embeddings)],
        "fixed_anchor_sha256": fixed_anchor_sha256,
        "anchor_embeddings": [] if (
            reference_anchor_embeddings is None or
            current_anchor_embeddings is None) else [
                _array_hash(reference_anchor_embeddings),
                _array_hash(current_anchor_embeddings),
            ],
    }
    source_sha = _sha256(source_payload)
    payload = {
        "schema_version": 1,
        "report_version": spec.report_version,
        "monitor_spec_sha256": spec.sha256,
        "reference_window": _json_value(reference_window),
        "current_window": _json_value(current_window),
        "data_quality": data_quality,
        "raw_drift": raw_summary,
        "feature_drift": feature_summary,
        "feature_joint_drift": feature_joint,
        "prediction_drift": prediction_summary,
        "label_status": label_status.value,
        "label_drift": label_summary,
        "concept_status": concept_status.value,
        "concept_performance": concept_summary,
        "embedding_status": embedding_status.value,
        "embedding_drift": embedding_summary,
        "alert_state": state.value,
        "alert_reasons": reason_tuple,
        "persistence_count": persistence,
        "retraining_review_recommended": state is AlertState.CRITICAL,
        "artifact_hashes": normalized_hashes,
        "source_snapshot_set_sha256": source_sha,
    }
    report_sha = _sha256(payload)
    return DriftReport(
        schema_version=1,
        report_version=spec.report_version,
        monitor_spec_sha256=spec.sha256,
        reference_window=reference_window,
        current_window=current_window,
        data_quality=_freeze(data_quality),
        raw_drift=_freeze(raw_summary),
        feature_drift=_freeze(feature_summary),
        feature_joint_drift=_freeze(feature_joint),
        prediction_drift=_freeze(prediction_summary),
        label_status=label_status,
        label_drift=_freeze(label_summary),
        concept_status=concept_status,
        concept_performance=_freeze(concept_summary),
        embedding_status=embedding_status,
        embedding_drift=_freeze(embedding_summary),
        alert_state=state,
        alert_reasons=reason_tuple,
        persistence_count=persistence,
        retraining_review_recommended=state is AlertState.CRITICAL,
        artifact_hashes=_freeze(normalized_hashes),
        source_snapshot_set_sha256=source_sha,
        report_sha256=report_sha,
    )
