"""成本档案回归测试:印花税 point-in-time、最低佣金、零成本基线的纯粹性。

背景:虚拟盘此前用引擎默认 ExecutionConfig,FeeSchedule 完全没登记,所有会话
都是零成本。接上成本后必须锁住三件事,否则"扣了成本"的结论会失真:
  1. frictionless 必须真的为零(曾因继承默认印花税而变成 5bp);
  2. 印花税必须按 2023-08-28 切换,不能整段用单一费率;
  3. 最低佣金按每笔委托生效,不能在小额单上被忽略。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python.broker import STAMP_TAX_REDUCTION_NS  # noqa: E402
from python.paper import costs  # noqa: E402


def test_frictionless_profile_is_actually_zero():
    profile = costs.get_profile("frictionless")
    assert profile.round_trip_bps() == 0.0
    for rate in profile.schedules():
        assert rate.commission_rate == 0.0
        assert rate.min_commission == 0.0
        assert rate.stamp_tax_rate == 0.0
        assert rate.transfer_fee_rate == 0.0


def test_none_profile_matches_frictionless():
    assert costs.get_profile(None).name == "frictionless"


def test_stamp_tax_is_point_in_time():
    """跨 2023-08-28 的费率表必须两段,且费率各自正确。"""
    profile = costs.get_profile("ashare-w1")
    schedules = profile.schedules()
    assert len(schedules) == 2
    before, after = schedules
    assert before.effective_from == 0
    assert before.effective_to == STAMP_TAX_REDUCTION_NS
    assert before.stamp_tax_rate == costs.STAMP_TAX_BEFORE_2023_08_28
    assert after.effective_from == STAMP_TAX_REDUCTION_NS
    assert after.effective_to is None
    assert after.stamp_tax_rate == costs.STAMP_TAX_FROM_2023_08_28


def test_explicit_stamp_tax_collapses_to_single_segment():
    profile = costs.CostProfile(
        name="t", note="", commission_rate=0.0001, stamp_tax_rate=0.0
    )
    assert len(profile.schedules()) == 1
    assert profile.effective_stamp_tax() == 0.0


def test_stamp_tax_reduction_constant_is_2023_08_28():
    import datetime as dt

    moment = dt.datetime.fromtimestamp(
        STAMP_TAX_REDUCTION_NS / 1e9, dt.timezone.utc
    )
    assert moment.date() == dt.date(2023, 8, 27)  # 北京时间 2023-08-28
    assert (moment + dt.timedelta(hours=8)).date() == dt.date(2023, 8, 28)


def test_round_trip_bps_matches_rate_arithmetic():
    profile = costs.get_profile("ashare-w1")
    expected = (0.0001 + 0.00001) * 2 * 10_000 + costs.STAMP_TAX_FROM_2023_08_28 * 10_000
    assert profile.round_trip_bps() == pytest.approx(expected)
    assert profile.round_trip_bps() == pytest.approx(7.2)


@pytest.mark.parametrize(
    "notional,is_sell,expected",
    [
        # 大额:佣金按费率,无最低佣金介入
        (100_000.0, True, 10.0 + 1.0 + 50.0),
        (100_000.0, False, 10.0 + 1.0),
        # 小额:佣金被 5 元下限托底
        (10_000.0, True, 5.0 + 0.1 + 5.0),
        (10_000.0, False, 5.0 + 0.1),
    ],
)
def test_fee_arithmetic_matches_engine_formula(notional, is_sell, expected):
    """复刻引擎 calculate_commission 的公式,确认登记值算出来的费用符合预期。"""
    profile = costs.get_profile("ashare-w1")
    commission = max(notional * profile.commission_rate, profile.min_commission)
    total = commission + notional * profile.transfer_fee_rate
    if is_sell:
        total += notional * costs.STAMP_TAX_FROM_2023_08_28
    assert total == pytest.approx(expected, abs=1e-6)


def test_unknown_profile_rejected():
    with pytest.raises(ValueError):
        costs.get_profile("no-such-profile")
