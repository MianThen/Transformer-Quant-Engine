from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np
import pandas as pd


class RankingScoreMode(str, Enum):
    RAW_RETURN = "raw_return"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"


@dataclass(frozen=True)
class RankingScoreSpecV1:
    mode: RankingScoreMode = RankingScoreMode.RAW_RETURN
    production_top_k: int = 20
    risk_floor: float = 1e-4
    cost_proxy_bps: float = 0.0
    winsor_lower_quantile: float = 0.01
    winsor_upper_quantile: float = 0.99
    rank_temperature: float = 1.0
    lambda_rank: float = 0.1
    target_tie_policy: str = "symbol_ascending"

    def __post_init__(self) -> None:
        if not isinstance(self.mode, RankingScoreMode):
            object.__setattr__(self, "mode", RankingScoreMode(self.mode))
        values = (
            self.risk_floor,
            self.cost_proxy_bps,
            self.winsor_lower_quantile,
            self.winsor_upper_quantile,
            self.rank_temperature,
            self.lambda_rank,
        )
        if self.production_top_k <= 0 or not np.isfinite(values).all():
            raise ValueError("RankingScoreSpec 数值必须有限且 production_top_k 为正数")
        if self.risk_floor <= 0 or self.cost_proxy_bps < 0:
            raise ValueError("risk_floor 必须为正，cost_proxy_bps 不能为负")
        if not 0 <= self.winsor_lower_quantile < self.winsor_upper_quantile <= 1:
            raise ValueError("winsor 分位点必须满足 0 <= lower < upper <= 1")
        if self.rank_temperature <= 0 or self.lambda_rank < 0:
            raise ValueError("rank_temperature 必须为正，lambda_rank 不能为负")
        if self.target_tie_policy != "symbol_ascending":
            raise ValueError("V1 只允许 symbol_ascending target tie policy")

    @property
    def canonical_json(self) -> str:
        value = asdict(self)
        value["mode"] = self.mode.value
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LabelSpecV2:
    horizon_bars: int = 5
    execution_lag_bars: int = 1
    direction_threshold: float = 0.0
    direction_temperature: float = 0.0025
    ranking: RankingScoreSpecV1 = RankingScoreSpecV1()

    def __post_init__(self) -> None:
        if self.horizon_bars <= 0 or self.execution_lag_bars <= 0:
            raise ValueError("horizon_bars 和 execution_lag_bars 必须为正数")
        if not np.isfinite(self.direction_threshold):
            raise ValueError("direction_threshold 必须有限")
        if not np.isfinite(self.direction_temperature) or self.direction_temperature <= 0:
            raise ValueError("direction_temperature 必须为有限正数")

    @property
    def canonical_json(self) -> str:
        value = asdict(self)
        value["ranking"]["mode"] = self.ranking.mode.value
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


def production_ranking_score(
    expected_return,
    expected_volatility,
    spec: RankingScoreSpecV1,
):
    returns = np.asarray(expected_return)
    volatility = np.asarray(expected_volatility)
    if returns.shape != volatility.shape:
        raise ValueError("收益和波动预测 shape 必须一致")
    if not np.isfinite(returns).all() or not np.isfinite(volatility).all():
        raise ValueError("ranking score 输入必须有限")
    if (volatility < 0).any():
        raise ValueError("预测波动不能为负")
    if spec.mode is RankingScoreMode.RAW_RETURN:
        return returns.copy()
    return returns / np.maximum(volatility, spec.risk_floor)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _rank_relevance(values: pd.Series) -> pd.Series:
    valid = values.notna()
    result = pd.Series(np.nan, index=values.index, dtype=np.float64)
    count = int(valid.sum())
    if count == 0:
        return result
    if count == 1:
        result.loc[valid] = 0.5
        return result
    ranks = values.loc[valid].rank(method="average", ascending=True)
    result.loc[valid] = (ranks - 1.0) / (count - 1.0)
    return result


def build_label_v2(source: pd.DataFrame, spec: LabelSpecV2) -> pd.DataFrame:
    required = {"timestamp", "symbol", "open", "close"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError("LabelSpec V2 缺少字段: " + ", ".join(sorted(missing)))
    table = source.copy().sort_values(["symbol", "timestamp"], kind="stable")
    if table.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("LabelSpec V2 不允许重复 timestamp/symbol")
    if not np.isfinite(table[["open", "close"]].to_numpy(dtype=np.float64)).all():
        raise ValueError("LabelSpec V2 价格必须有限")
    if (table[["open", "close"]] <= 0).any().any():
        raise ValueError("LabelSpec V2 价格必须为正")

    records: list[dict] = []
    exit_offset = spec.execution_lag_bars + spec.horizon_bars
    for symbol, group in table.groupby("symbol", sort=False):
        group = group.reset_index(drop=True)
        opens = group["open"].to_numpy(dtype=np.float64)
        closes = group["close"].to_numpy(dtype=np.float64)
        for index, row in group.iterrows():
            entry_index = index + spec.execution_lag_bars
            exit_index = index + exit_offset
            record = {"timestamp": row["timestamp"], "symbol": symbol}
            if exit_index >= len(group):
                record.update({
                    "entry_open": np.nan,
                    "exit_close": np.nan,
                    "return_raw": np.nan,
                    "expected_return": np.nan,
                    "direction_soft": np.nan,
                    "direction": np.nan,
                    "realized_volatility": np.nan,
                    "downside_semivol": np.nan,
                    "risk_adjusted_return": np.nan,
                    "label_valid": False,
                })
                records.append(record)
                continue
            entry = opens[entry_index]
            exit_price = closes[exit_index]
            subreturns = [np.log(closes[entry_index] / entry)]
            subreturns.extend(
                np.log(closes[position] / closes[position - 1])
                for position in range(entry_index + 1, exit_index + 1)
            )
            subreturns_array = np.asarray(subreturns, dtype=np.float64)
            raw_return = float(np.log(exit_price / entry))
            volatility = float(np.std(subreturns_array, ddof=0))
            downside = float(np.sqrt(np.mean(np.minimum(subreturns_array, 0.0) ** 2)))
            risk_adjusted = raw_return / max(volatility, spec.ranking.risk_floor)
            soft = float(_sigmoid(np.asarray([
                (raw_return - spec.direction_threshold) /
                spec.direction_temperature
            ]))[0])
            record.update({
                "entry_open": entry,
                "exit_close": exit_price,
                "return_raw": raw_return,
                "expected_return": raw_return,
                "direction_soft": soft,
                "direction": soft,
                "realized_volatility": volatility,
                "downside_semivol": downside,
                "risk_adjusted_return": risk_adjusted,
                "label_valid": True,
            })
            records.append(record)

    result = pd.DataFrame.from_records(records)
    cost = spec.ranking.cost_proxy_bps * 1e-4
    utility = result["return_raw"] - cost
    if spec.ranking.mode is RankingScoreMode.RISK_ADJUSTED_RETURN:
        utility = utility / np.maximum(
            result["realized_volatility"], spec.ranking.risk_floor
        )
    result["rank_utility_unwinsorized"] = utility

    def winsorize(group: pd.Series) -> pd.Series:
        valid = group.dropna()
        if valid.empty:
            return group
        lower = valid.quantile(spec.ranking.winsor_lower_quantile)
        upper = valid.quantile(spec.ranking.winsor_upper_quantile)
        return group.clip(lower, upper)

    result["rank_utility"] = result.groupby("timestamp", sort=False)[
        "rank_utility_unwinsorized"
    ].transform(winsorize)
    result["rank_relevance"] = result.groupby("timestamp", sort=False)[
        "rank_utility"
    ].transform(_rank_relevance)
    result.loc[~result["label_valid"], [
        "rank_utility_unwinsorized", "rank_utility", "rank_relevance"
    ]] = np.nan
    result["label_spec_sha256"] = spec.sha256
    result["ranking_score_spec_sha256"] = spec.ranking.sha256
    return result.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)
