from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class RankingLossDiagnostics:
    cross_sections: int = 0
    contributing_cross_sections: int = 0
    idcg_zero_cross_sections: int = 0
    pair_count: int = 0


@dataclass(frozen=True)
class RankingVariantComparison:
    input_fingerprint: str
    cross_sections: int
    legacy_loss: float
    listmle_loss: float
    lambda_loss: float
    ndcg_at_cutoff: float


@dataclass(frozen=True)
class RankingOOSMetrics:
    cross_sections: int
    transition_count: int
    ndcg_at_cutoff: float
    rank_ic: float
    precision_at_cutoff: float
    top_k_overlap: float
    top_k_turnover: float
    top_bottom_utility_spread: float


class CrossSectionBatchSampler:
    """Yield complete timestamp lists; a timestamp is never split across batches."""

    def __init__(
        self,
        timestamps,
        *,
        cross_sections_per_batch: int = 1,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        values = np.asarray(timestamps)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("timestamps 必须是一维非空数组")
        if cross_sections_per_batch <= 0:
            raise ValueError("cross_sections_per_batch 必须为正数")
        self._groups = [np.flatnonzero(values == value).tolist() for value in np.unique(values)]
        self._cross_sections_per_batch = cross_sections_per_batch
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0

    def __iter__(self):
        order = np.arange(len(self._groups))
        if self._shuffle:
            np.random.default_rng(self._seed + self._epoch).shuffle(order)
        self._epoch += 1
        for start in range(0, order.size, self._cross_sections_per_batch):
            batch: list[int] = []
            for index in order[start:start + self._cross_sections_per_batch]:
                batch.extend(self._groups[int(index)])
            yield batch

    def __len__(self) -> int:
        return (len(self._groups) + self._cross_sections_per_batch - 1) // self._cross_sections_per_batch


def _validate_numpy(scores, relevance, mask, cutoff):
    scores = np.asarray(scores, dtype=np.float64)
    relevance = np.asarray(relevance, dtype=np.float64)
    if scores.ndim != 1 or relevance.shape != scores.shape:
        raise ValueError("scores/relevance 必须是一维同 shape")
    mask = np.ones(scores.size, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != scores.shape or cutoff <= 0:
        raise ValueError("mask shape 必须一致且 cutoff 为正数")
    if not np.isfinite(scores[mask]).all() or not np.isfinite(relevance[mask]).all():
        raise ValueError("有效 ranking 输入必须有限")
    if ((relevance[mask] < 0) | (relevance[mask] > 1)).any():
        raise ValueError("relevance 必须位于 [0,1]")
    return scores, relevance, mask


def _average_tie_discounts(scores: np.ndarray, cutoff: int) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    discounts_by_position = np.zeros(scores.size, dtype=np.float64)
    positions = np.arange(1, scores.size + 1)
    inside = positions <= cutoff
    discounts_by_position[inside] = 1.0 / np.log2(positions[inside] + 1.0)
    result = np.zeros(scores.size, dtype=np.float64)
    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and scores[order[end]] == scores[order[start]]:
            end += 1
        result[order[start:end]] = discounts_by_position[start:end].mean()
        start = end
    return result


def ndcg_at_k(scores, relevance, *, cutoff: int, mask=None, epsilon: float = 1e-12) -> float:
    scores, relevance, mask = _validate_numpy(scores, relevance, mask, cutoff)
    scores, relevance = scores[mask], relevance[mask]
    if scores.size == 0:
        return 0.0
    gains = np.exp2(relevance) - 1.0
    dcg = float(np.dot(gains, _average_tie_discounts(scores, cutoff)))
    ideal = np.sort(gains)[::-1]
    positions = np.arange(1, ideal.size + 1)
    discounts = np.where(
        positions <= cutoff, 1.0 / np.log2(positions + 1.0), 0.0
    )
    idcg = float(np.dot(ideal, discounts))
    return 0.0 if idcg <= epsilon else dcg / idcg


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _rank_ic(scores: np.ndarray, utility: np.ndarray) -> float:
    if scores.size < 2:
        return 0.0
    score_rank = _average_ranks(scores)
    utility_rank = _average_ranks(utility)
    if np.std(score_rank) == 0 or np.std(utility_rank) == 0:
        return 0.0
    return float(np.corrcoef(score_rank, utility_rank)[0, 1])


def ranking_oos_metrics(
    scores,
    utility,
    relevance,
    timestamps,
    symbols,
    *,
    cutoff: int,
    mask=None,
) -> RankingOOSMetrics:
    """Timestamp-equal OOS metrics; top-k ties use frozen symbol ascending order."""
    scores = np.asarray(scores, dtype=np.float64)
    utility = np.asarray(utility, dtype=np.float64)
    relevance = np.asarray(relevance, dtype=np.float64)
    timestamps = np.asarray(timestamps)
    symbols = np.asarray(symbols).astype(str)
    if scores.ndim != 1 or not all(value.shape == scores.shape for value in (
        utility, relevance, timestamps, symbols
    )):
        raise ValueError("OOS ranking metric 输入必须是一维同 shape")
    mask = np.ones(scores.size, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != scores.shape or cutoff <= 0:
        raise ValueError("OOS ranking metric mask/cutoff 无效")
    if not (np.isfinite(scores[mask]).all() and np.isfinite(utility[mask]).all()
            and np.isfinite(relevance[mask]).all()):
        raise ValueError("OOS ranking metric 有效输入必须有限")

    ndcg_values: list[float] = []
    rank_ic_values: list[float] = []
    precision_values: list[float] = []
    spread_values: list[float] = []
    top_sets: list[set[str]] = []
    for timestamp in np.unique(timestamps):
        selected = (timestamps == timestamp) & mask
        group_scores = scores[selected]
        group_utility = utility[selected]
        group_relevance = relevance[selected]
        group_symbols = symbols[selected]
        if group_scores.size == 0:
            continue
        k = min(cutoff, group_scores.size)
        predicted = np.lexsort((group_symbols, -group_scores))
        ideal = np.lexsort((group_symbols, -group_utility))
        predicted_top = set(group_symbols[predicted[:k]].tolist())
        ideal_top = set(group_symbols[ideal[:k]].tolist())
        ndcg_values.append(ndcg_at_k(group_scores, group_relevance, cutoff=cutoff))
        rank_ic_values.append(_rank_ic(group_scores, group_utility))
        precision_values.append(len(predicted_top & ideal_top) / k)
        spread_values.append(float(
            group_utility[predicted[:k]].mean() - group_utility[predicted[-k:]].mean()
        ))
        top_sets.append(predicted_top)
    if not ndcg_values:
        raise ValueError("OOS ranking metric 没有有效 timestamp")
    overlaps = [
        len(previous & current) / max(1, min(len(previous), len(current)))
        for previous, current in zip(top_sets, top_sets[1:])
    ]
    overlap = float(np.mean(overlaps)) if overlaps else 1.0
    return RankingOOSMetrics(
        cross_sections=len(ndcg_values),
        transition_count=len(overlaps),
        ndcg_at_cutoff=float(np.mean(ndcg_values)),
        rank_ic=float(np.mean(rank_ic_values)),
        precision_at_cutoff=float(np.mean(precision_values)),
        top_k_overlap=overlap,
        top_k_turnover=1.0 - overlap,
        top_bottom_utility_spread=float(np.mean(spread_values)),
    )


def lambda_loss_numpy(
    scores,
    relevance,
    *,
    cutoff: int,
    temperature: float = 1.0,
    mask=None,
    optimized_topk: bool = True,
    epsilon: float = 1e-12,
) -> tuple[float, RankingLossDiagnostics]:
    scores, relevance, mask = _validate_numpy(scores, relevance, mask, cutoff)
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature 必须为有限正数")
    scores, relevance = scores[mask], relevance[mask]
    count = scores.size
    if count < 2:
        return 0.0, RankingLossDiagnostics(1, 0, 1, 0)
    gains = np.exp2(relevance) - 1.0
    ideal = np.sort(gains)[::-1]
    positions = np.arange(1, count + 1)
    ideal_discount = np.where(
        positions <= cutoff, 1.0 / np.log2(positions + 1.0), 0.0
    )
    idcg = float(np.dot(ideal, ideal_discount))
    if idcg <= epsilon:
        return 0.0, RankingLossDiagnostics(1, 0, 1, 0)
    discounts = _average_tie_discounts(scores, cutoff)
    anchors = np.flatnonzero(discounts > 0) if optimized_topk else np.arange(count)
    weighted_loss = 0.0
    weight_sum = 0.0
    pair_count = 0
    seen: set[tuple[int, int]] = set()
    for first in anchors:
        for second in range(count):
            if first == second or relevance[first] == relevance[second]:
                continue
            high, low = (first, second) if relevance[first] > relevance[second] else (second, first)
            pair = (int(high), int(low))
            if pair in seen:
                continue
            seen.add(pair)
            weight = abs(gains[high] - gains[low]) * abs(
                discounts[high] - discounts[low]
            ) / idcg
            if weight <= 0:
                continue
            margin = (scores[high] - scores[low]) / temperature
            weighted_loss += weight * float(np.logaddexp(0.0, -margin))
            weight_sum += weight
            pair_count += 1
    loss = 0.0 if weight_sum <= epsilon else weighted_loss / weight_sum
    return loss, RankingLossDiagnostics(1, int(weight_sum > epsilon), 0, pair_count)


def listmle_loss_numpy(scores, utility, *, mask=None, tie_breaker=None) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    utility = np.asarray(utility, dtype=np.float64)
    if scores.ndim != 1 or utility.shape != scores.shape:
        raise ValueError("scores/utility 必须是一维同 shape")
    mask = np.ones(scores.size, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != scores.shape or not np.isfinite(scores[mask]).all() or not np.isfinite(utility[mask]).all():
        raise ValueError("ListMLE mask 或有效输入无效")
    scores, utility = scores[mask], utility[mask]
    if tie_breaker is None:
        if np.unique(utility).size != utility.size:
            raise ValueError("ListMLE target tie 必须提供冻结的 tie_breaker")
        tie_breaker = np.arange(scores.size)
    else:
        tie_breaker = np.asarray(tie_breaker)[mask]
        if np.unique(tie_breaker).size != tie_breaker.size:
            raise ValueError("tie_breaker 必须唯一")
    if scores.size < 2:
        return 0.0
    order = np.lexsort((tie_breaker, -utility))
    ordered = scores[order]
    loss = 0.0
    for index in range(ordered.size):
        tail = ordered[index:]
        maximum = tail.max()
        loss += maximum + np.log(np.exp(tail - maximum).sum()) - ordered[index]
    return float(loss)


def compare_ranking_variants_numpy(
    scores,
    utility,
    relevance,
    timestamps,
    tie_breaker,
    *,
    cutoff: int,
    temperature: float,
    mask=None,
) -> RankingVariantComparison:
    scores = np.asarray(scores, dtype=np.float64)
    utility = np.asarray(utility, dtype=np.float64)
    relevance = np.asarray(relevance, dtype=np.float64)
    timestamps = np.asarray(timestamps)
    tie_breaker = np.asarray(tie_breaker)
    if not all(value.shape == scores.shape for value in (
        utility, relevance, timestamps, tie_breaker
    )):
        raise ValueError("ranking paired report 输入 shape 必须一致")
    mask = np.ones(scores.size, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != scores.shape:
        raise ValueError("ranking paired report mask shape 无效")
    digest = hashlib.sha256()
    for values in (scores, utility, relevance, timestamps, tie_breaker, mask):
        digest.update(np.ascontiguousarray(values).tobytes())

    legacy_values = []
    listmle_values = []
    lambda_values = []
    ndcg_values = []
    for timestamp in np.unique(timestamps):
        selected = (timestamps == timestamp) & mask
        group_scores = scores[selected]
        group_utility = utility[selected]
        group_relevance = relevance[selected]
        group_ties = tie_breaker[selected]
        if group_scores.size == 0:
            continue
        if group_scores.size < 2 or np.std(group_scores) == 0 or np.std(group_utility) == 0:
            legacy = 0.0
        else:
            legacy = -float(np.corrcoef(group_scores, group_utility)[0, 1])
        legacy_values.append(legacy)
        listmle_values.append(listmle_loss_numpy(
            group_scores, group_utility, tie_breaker=group_ties
        ))
        lambda_values.append(lambda_loss_numpy(
            group_scores,
            group_relevance,
            cutoff=cutoff,
            temperature=temperature,
        )[0])
        ndcg_values.append(ndcg_at_k(
            group_scores, group_relevance, cutoff=cutoff
        ))
    if not legacy_values:
        raise ValueError("ranking paired report 没有有效 timestamp")
    return RankingVariantComparison(
        input_fingerprint=digest.hexdigest(),
        cross_sections=len(legacy_values),
        legacy_loss=float(np.mean(legacy_values)),
        listmle_loss=float(np.mean(listmle_values)),
        lambda_loss=float(np.mean(lambda_values)),
        ndcg_at_cutoff=float(np.mean(ndcg_values)),
    )


def _torch():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("ranking autograd 需要安装项目 ml 可选依赖") from exc
    return torch, functional


def legacy_rank_loss(scores, utility, mask=None):
    torch, _ = _torch()
    mask = torch.ones_like(scores, dtype=torch.bool) if mask is None else mask.bool()
    selected_scores = scores[mask]
    selected_utility = utility[mask]
    if selected_scores.numel() < 2:
        return scores.sum() * 0.0
    centered_scores = selected_scores - selected_scores.mean()
    centered_utility = selected_utility - selected_utility.mean()
    denominator = torch.sqrt(
        centered_scores.square().sum() * centered_utility.square().sum()
    )
    if float(denominator.detach()) <= torch.finfo(scores.dtype).eps:
        return scores.sum() * 0.0
    return -(centered_scores * centered_utility).sum() / denominator


def listmle_loss(scores, utility, *, mask=None, tie_breaker=None):
    torch, _ = _torch()
    mask = torch.ones_like(scores, dtype=torch.bool) if mask is None else mask.bool()
    scores = scores[mask]
    utility = utility[mask]
    if tie_breaker is None:
        tie_breaker = torch.arange(scores.numel(), device=scores.device)
        if torch.unique(utility).numel() != utility.numel():
            raise ValueError("ListMLE target tie 必须提供冻结的 tie_breaker")
    else:
        tie_breaker = tie_breaker[mask]
    if scores.numel() < 2:
        return scores.sum() * 0.0
    secondary = torch.argsort(tie_breaker, stable=True)
    primary = torch.argsort(utility[secondary], descending=True, stable=True)
    ordered_scores = scores[secondary[primary]]
    denominators = torch.logcumsumexp(ordered_scores.flip(0), dim=0).flip(0)
    return (denominators - ordered_scores).sum()


def lambda_loss_at_k(
    scores,
    relevance,
    *,
    cutoff: int,
    temperature: float = 1.0,
    mask=None,
    epsilon: float = 1e-12,
):
    torch, functional = _torch()
    if cutoff <= 0 or temperature <= 0:
        raise ValueError("cutoff 和 temperature 必须为正数")
    mask = torch.ones_like(scores, dtype=torch.bool) if mask is None else mask.bool()
    scores = scores[mask]
    relevance = relevance[mask]
    count = scores.numel()
    if count < 2:
        return scores.sum() * 0.0
    with torch.no_grad():
        gains = torch.exp2(relevance) - 1.0
        ideal = torch.sort(gains, descending=True).values
        positions = torch.arange(1, count + 1, device=scores.device, dtype=scores.dtype)
        ideal_discount = torch.where(
            positions <= cutoff, 1.0 / torch.log2(positions + 1.0),
            torch.zeros_like(positions),
        )
        idcg = (ideal * ideal_discount).sum()
        if float(idcg) <= epsilon:
            return scores.sum() * 0.0
        order = torch.argsort(scores, descending=True, stable=True)
        position_discount = ideal_discount
        discounts = torch.zeros_like(scores)
        start = 0
        while start < count:
            end = start + 1
            while end < count and bool(scores[order[end]] == scores[order[start]]):
                end += 1
            discounts[order[start:end]] = position_discount[start:end].mean()
            start = end
        anchors = torch.nonzero(discounts > 0, as_tuple=False).flatten()

    first = anchors.repeat_interleave(count)
    second = torch.arange(count, device=scores.device).repeat(anchors.numel())
    usable = (first != second) & (relevance[first] != relevance[second])
    first, second = first[usable], second[usable]
    high = torch.where(relevance[first] > relevance[second], first, second)
    low = torch.where(relevance[first] > relevance[second], second, first)
    pair_keys = torch.unique(high * count + low)
    high, low = pair_keys // count, pair_keys % count
    with torch.no_grad():
        weights = torch.abs(gains[high] - gains[low]) * torch.abs(
            discounts[high] - discounts[low]
        ) / idcg
        positive = weights > 0
    high, low, weights = high[positive], low[positive], weights[positive]
    if weights.numel() == 0:
        return scores.sum() * 0.0
    weighted_losses = weights * functional.softplus(
        -(scores[high] - scores[low]) / temperature
    )
    return weighted_losses.sum() / weights.sum().clamp_min(epsilon)
