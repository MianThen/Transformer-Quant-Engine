"""模拟券商:手续费、印花税、滑点。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine_api import Order


STAMP_TAX_REDUCTION_NS = 1_693_152_000_000_000_000


@dataclass(frozen=True)
class FeeSchedule:
    effective_from: int
    effective_to: int | None
    commission_rate: float
    min_commission: float
    stamp_tax_rate: float
    transfer_fee_rate: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.commission_rate, self.min_commission,
            self.stamp_tax_rate, self.transfer_fee_rate,
        )
        if self.effective_from < 0 or any(
            not math.isfinite(value) or value < 0.0 for value in values
        ):
            raise ValueError("非法费率配置")
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("effective_to 必须晚于 effective_from")


class Broker:
    """交易成本模型。

    A 股默认规则:
      - 佣金: 双边 万三,最低 5 元
      - 印花税: 2023-08-28 前卖出千一，之后卖出万五
      - 过户费: 默认 0，可通过 point-in-time FeeSchedule 配置
    """

    def __init__(
        self,
        commission_rate: float = 0.0003,
        min_commission: float = 5.0,
        stamp_tax_rate: float | None = None,
        transfer_fee_rate: float = 0.0,
        slippage: float = 0.0,
        fee_schedules: list[FeeSchedule] | None = None,
    ):
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax_rate = stamp_tax_rate
        self.transfer_fee_rate = transfer_fee_rate
        self.slippage = slippage
        if not math.isfinite(slippage) or not 0.0 <= slippage < 1.0:
            raise ValueError("slippage 必须在 [0, 1) 内")
        if fee_schedules is not None:
            schedules = list(fee_schedules)
        elif stamp_tax_rate is None:
            schedules = [
                FeeSchedule(0, STAMP_TAX_REDUCTION_NS, commission_rate,
                            min_commission, 0.001, transfer_fee_rate),
                FeeSchedule(STAMP_TAX_REDUCTION_NS, None, commission_rate,
                            min_commission, 0.0005, transfer_fee_rate),
            ]
        else:
            schedules = [FeeSchedule(
                0, None, commission_rate, min_commission,
                stamp_tax_rate, transfer_fee_rate,
            )]
        self._schedules = tuple(sorted(schedules, key=lambda item: item.effective_from))
        if not self._schedules:
            raise ValueError("fee_schedules 不能为空")
        for previous, current in zip(self._schedules, self._schedules[1:]):
            if previous.effective_to is None or previous.effective_to > current.effective_from:
                raise ValueError("费率生效区间重叠")
        self._timestamp = 0

    def set_timestamp(self, timestamp: int) -> None:
        if timestamp < 0:
            raise ValueError("timestamp 不能为负数")
        self._timestamp = timestamp

    @property
    def fee_schedules(self) -> tuple[FeeSchedule, ...]:
        return self._schedules

    def commission(self, notional: float, is_sell: bool) -> float:
        """计算一笔成交的总费用(佣金 + 卖出印花税)。"""
        return self.commission_at(self._timestamp, notional, is_sell)

    def to_config(self) -> dict[str, object]:
        return {
            "commission_rate": self.commission_rate,
            "min_commission": self.min_commission,
            "stamp_tax_rate": self.stamp_tax_rate,
            "transfer_fee_rate": self.transfer_fee_rate,
            "slippage": self.slippage,
            "fee_schedules": [asdict(schedule) for schedule in self._schedules],
        }

    def commission_at(self, timestamp: int, notional: float, is_sell: bool) -> float:
        if not math.isfinite(notional) or notional < 0.0:
            raise ValueError("notional 必须是有限非负数")
        schedule = self._schedule_at(timestamp)
        fee = max(notional * schedule.commission_rate, schedule.min_commission)
        fee += notional * schedule.transfer_fee_rate
        if is_sell:
            fee += notional * schedule.stamp_tax_rate
        return fee

    def _schedule_at(self, timestamp: int) -> FeeSchedule:
        for schedule in reversed(self._schedules):
            if timestamp < schedule.effective_from:
                continue
            if schedule.effective_to is None or timestamp < schedule.effective_to:
                return schedule
        raise ValueError(f"timestamp={timestamp} 没有可用费率")

    def apply_slippage(self, price: float, is_buy: bool) -> float:
        """按滑点调整成交价(买入抬价、卖出压价)。"""
        if is_buy:
            return price * (1 + self.slippage)
        return price * (1 - self.slippage)
