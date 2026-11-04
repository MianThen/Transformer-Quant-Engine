"""虚拟盘交易成本档案:把 A 股市场税费与滑点显式登记为回放输入。

费率口径与 `python/broker.py`(Python 回测路径)保持一致,本模块负责把它翻译成
C++ 引擎的 `FeeSchedule` / `ExecutionConfig` 可调用形式,避免两条路径口径分叉。

设计约束:
  - 费率是研究输入,不是隐含默认值。每个会话必须显式绑定一个 profile 名,
    并在 paper.db 的 sessions.cost_profile 中留痕,便于事后审计"这个收益扣了多少成本"。
  - 历史费率必须 point-in-time:2023-08-28 印花税由千一下调为万五,跨该日的回放
    不能只用单一费率。
  - 最低佣金按"每笔委托"计,与引擎 calculate_commission 的
    `max(notional * commission_rate, min_commission)` 语义一致。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from python.broker import STAMP_TAX_REDUCTION_NS


# 印花税单边(仅卖出)费率,2023-08-28 起由千一 → 万五
STAMP_TAX_BEFORE_2023_08_28 = 0.001
STAMP_TAX_FROM_2023_08_28 = 0.0005

# 中国结算过户费(沪深统一,双向):成交金额的 0.001%
TRANSFER_FEE_RATE = 0.00001

# 研究与实盘必须共享的最小佣金(元)
DEFAULT_MIN_COMMISSION = 5.0


@dataclass(frozen=True)
class FeeRate:
    """一段有效期内的费率。effective_to 为 None 表示持续有效。"""

    effective_from: int
    effective_to: int | None
    commission_rate: float
    min_commission: float = DEFAULT_MIN_COMMISSION
    stamp_tax_rate: float = STAMP_TAX_FROM_2023_08_28
    transfer_fee_rate: float = TRANSFER_FEE_RATE

    def as_dict(self) -> dict:
        return {
            "effective_from": self.effective_from,
            "effective_to": self.effective_to,
            "commission_rate": self.commission_rate,
            "min_commission": self.min_commission,
            "stamp_tax_rate": self.stamp_tax_rate,
            "transfer_fee_rate": self.transfer_fee_rate,
        }


@dataclass(frozen=True)
class CostProfile:
    """一个可复现的成本假设集合。

    stamp_tax_rate 为 None 表示按 point-in-time 历史费率(2023-08-28 前后不同);
    显式给值则整段用该固定费率——零成本基线必须显式传 0.0。
    """

    name: str
    note: str
    commission_rate: float
    min_commission: float = DEFAULT_MIN_COMMISSION
    transfer_fee_rate: float = TRANSFER_FEE_RATE
    slippage_bps: float = 0.0
    max_volume_participation: float = 0.10
    stamp_tax_rate: float | None = None

    def _rate(self, effective_from: int, effective_to: int | None,
              stamp_tax: float) -> FeeRate:
        return FeeRate(
            effective_from=effective_from,
            effective_to=effective_to,
            commission_rate=self.commission_rate,
            min_commission=self.min_commission,
            stamp_tax_rate=stamp_tax,
            transfer_fee_rate=self.transfer_fee_rate,
        )

    def schedules(self) -> tuple[FeeRate, ...]:
        """按印花税生效日切分的费率表;显式指定费率时退化为单段。"""
        if self.stamp_tax_rate is not None:
            return (self._rate(0, None, self.stamp_tax_rate),)
        return (
            self._rate(0, STAMP_TAX_REDUCTION_NS, STAMP_TAX_BEFORE_2023_08_28),
            self._rate(STAMP_TAX_REDUCTION_NS, None, STAMP_TAX_FROM_2023_08_28),
        )

    def effective_stamp_tax(self) -> float:
        """当前(2023-08-28 之后)生效的印花税率,用于量级估算。"""
        if self.stamp_tax_rate is not None:
            return self.stamp_tax_rate
        return STAMP_TAX_FROM_2023_08_28

    def round_trip_bps(self) -> float:
        """一买一卖的显性成本(bp,不含滑点),用于快速判断成本量级。"""
        stamp = self.effective_stamp_tax()
        sell_bps = (self.commission_rate + self.transfer_fee_rate + stamp) * 10_000
        buy_bps = (self.commission_rate + self.transfer_fee_rate) * 10_000
        return buy_bps + sell_bps

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "note": self.note,
            "commission_rate": self.commission_rate,
            "min_commission": self.min_commission,
            "transfer_fee_rate": self.transfer_fee_rate,
            "stamp_tax_rate_from_2023_08_28": STAMP_TAX_FROM_2023_08_28,
            "stamp_tax_rate_before": STAMP_TAX_BEFORE_2023_08_28,
            "slippage_bps": self.slippage_bps,
            "max_volume_participation": self.max_volume_participation,
            "explicit_round_trip_bps": round(self.round_trip_bps(), 3),
        }


# 佣金费率区间很宽(取决于券商与账户),故登记三档而不是单一"默认值"。
# 研究结论必须同时给出所用档位;换券商账户时改登记值,不要改代码。
PROFILES: dict[str, CostProfile] = {
    "frictionless": CostProfile(
        name="frictionless",
        note="零成本基线:仅用于与历史结果对照,不得作为决策依据",
        commission_rate=0.0,
        min_commission=0.0,
        transfer_fee_rate=0.0,
        stamp_tax_rate=0.0,
    ),
    "ashare-w1": CostProfile(
        name="ashare-w1",
        note="互联网券商常见档:佣金万1(最低5元)+ 过户费0.001% + 印花税万五(卖出)",
        commission_rate=0.0001,
    ),
    "ashare-w3": CostProfile(
        name="ashare-w3",
        note="保守档:佣金万3(最低5元)+ 过户费0.001% + 印花税万五(卖出)",
        commission_rate=0.0003,
    ),
    "ashare-w1-spread": CostProfile(
        name="ashare-w1-spread",
        note="万1 显性费用 + 每边让 1 个最小报价单位滑点(14bp,低价股半档量级)",
        commission_rate=0.0001,
        slippage_bps=14.0,
    ),
}


def get_profile(name: str | None) -> CostProfile:
    """按名取成本档案;None 表示沿用历史零成本口径。"""
    if name is None:
        return PROFILES["frictionless"]
    try:
        return PROFILES[name]
    except KeyError:
        known = ", ".join(sorted(PROFILES))
        raise ValueError(f"未知成本档案 {name};可用: {known}") from None


def build_fee_schedules(module, profile: CostProfile) -> list:
    """翻译为 cpp_engine.FeeSchedule 列表(引擎要求按 effective_from 升序且不重叠)。"""
    return [
        module.FeeSchedule(
            rate.effective_from, rate.effective_to, rate.commission_rate,
            rate.min_commission, rate.stamp_tax_rate, rate.transfer_fee_rate,
        )
        for rate in profile.schedules()
    ]


def build_execution_config(module, profile: CostProfile):
    """按档案构造 ExecutionConfig(滑点与参与率上限一并登记)。"""
    config = module.ExecutionConfig()
    config.slippage_bps = profile.slippage_bps
    config.max_volume_participation = profile.max_volume_participation
    return config


def describe(profile: CostProfile) -> str:
    stamp_bp = profile.effective_stamp_tax() * 10_000
    stamp_note = ("point-in-time 千一→万五" if profile.stamp_tax_rate is None
                  else f"{stamp_bp:.1f}bp/卖出")
    return (f"{profile.name}: 显性往返 {profile.round_trip_bps():.1f}bp"
            f"(佣金 {profile.commission_rate*10_000:.1f}bp/边"
            f"{'、最低'+format(profile.min_commission,'g')+'元' if profile.min_commission else ''}"
            f"、过户费 {profile.transfer_fee_rate*10_000:.3f}bp/边"
            f"、印花税 {stamp_note})"
            f" + 滑点 {profile.slippage_bps:.1f}bp")
