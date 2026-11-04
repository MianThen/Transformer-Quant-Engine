from __future__ import annotations

import pytest

from python.broker import Broker, FeeSchedule, STAMP_TAX_REDUCTION_NS


def test_stamp_tax_is_point_in_time():
    broker = Broker()
    before = broker.commission_at(STAMP_TAX_REDUCTION_NS - 1, 10_000.0, True)
    after = broker.commission_at(STAMP_TAX_REDUCTION_NS, 10_000.0, True)
    assert before == pytest.approx(15.0)
    assert after == pytest.approx(10.0)
    assert broker.commission_at(STAMP_TAX_REDUCTION_NS, 10_000.0, False) == 5.0


def test_custom_fee_schedule_and_validation():
    broker = Broker(fee_schedules=[
        FeeSchedule(0, None, 0.0001, 0.0, 0.0, 0.00001),
    ])
    assert broker.commission_at(1, 10_000.0, False) == pytest.approx(1.1)
    with pytest.raises(ValueError, match="重叠"):
        Broker(fee_schedules=[
            FeeSchedule(0, None, 0.0, 0.0, 0.0),
            FeeSchedule(10, None, 0.0, 0.0, 0.0),
        ])
