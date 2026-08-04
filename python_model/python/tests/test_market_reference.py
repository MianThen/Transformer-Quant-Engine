from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from python.market_data.calendar import (
    BarTimestampConvention,
    CalendarValidationError,
    ChinaAShareCalendar,
    Session,
)
from python.market_data.reference import (
    AdjustmentFactor,
    AdjustmentFactorStore,
    CorporateAction,
    CorporateActionStore,
    MarketReferenceData,
    SecurityMaster,
    SecurityState,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")


def _ns(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=SHANGHAI).timestamp() * 1e9)


def test_explicit_calendar_rejects_holiday_lunch_and_wrong_convention():
    calendar = ChinaAShareCalendar(["2026-07-17"])
    assert calendar.session(_ns("2026-07-17T09:31:00")) == Session.CONTINUOUS
    assert calendar.session(_ns("2026-07-17T15:00:00")) == Session.CLOSING_AUCTION
    with pytest.raises(CalendarValidationError, match="非交易时间"):
        calendar.validate_timestamp(_ns("2026-07-17T12:00:00"))
    with pytest.raises(CalendarValidationError):
        calendar.validate_timestamp(_ns("2026-07-18T09:31:00"))

    start_calendar = ChinaAShareCalendar(
        ["2026-07-17"], timestamp_convention=BarTimestampConvention.START
    )
    assert start_calendar.is_trading_minute(_ns("2026-07-17T09:30:00"))
    assert not start_calendar.is_trading_minute(_ns("2026-07-17T11:30:00"))


def test_security_master_is_point_in_time_and_avoids_survivor_bias():
    master = SecurityMaster([
        SecurityState("OLD", 100, 200, board="MAIN", industry="Bank"),
        SecurityState("OLD", 200, 300, is_st=True, board="MAIN", industry="Bank"),
        SecurityState("NEW", 250, None, board="STAR", industry="Tech", lot_size=200),
    ])
    assert master.universe(150) == ("OLD",)
    assert master.universe(225) == ()
    assert master.universe(225, include_st=True) == ("OLD",)
    assert master.universe(275) == ("NEW",)
    assert master.at("NEW", 275).lot_size == 200
    assert master.at("OLD", 350) is None


def test_overlapping_temporal_intervals_are_rejected():
    with pytest.raises(ValueError, match="重叠"):
        SecurityMaster([
            SecurityState("X", 100, 300),
            SecurityState("X", 200, 400),
        ])
    with pytest.raises(ValueError, match="重叠"):
        SecurityMaster([
            SecurityState("X", 100, None),
            SecurityState("X", 200, 400),
        ])
    with pytest.raises(ValueError, match="重叠"):
        AdjustmentFactorStore([
            AdjustmentFactor("X", 100, 1.0, None),
            AdjustmentFactor("X", 200, 2.0, 400),
        ])


def test_adjusted_prices_are_signal_only_and_actions_are_ordered():
    reference = MarketReferenceData(
        security_master=SecurityMaster([
            SecurityState("X", 0, None, upper_limit=11.0, lower_limit=9.0,
                          board="MAIN", industry="Bank",
                          factor_exposures={"value": 0.7}),
        ]),
        adjustment_factors=AdjustmentFactorStore([
            AdjustmentFactor("X", 0, 0.5),
        ]),
        corporate_actions=CorporateActionStore([
            CorporateAction("X", 200, cash_dividend_per_share=0.2),
            CorporateAction("X", 300, share_multiplier=1.5),
        ]),
    )
    snapshot = SimpleNamespace(
        symbol="X", timestamp=100, open=10.0, high=12.0, low=9.0, close=11.0,
        is_listed=True, is_st=False, is_suspended=False, upper_limit=0.0,
        lower_limit=0.0, board="", industry="", lot_size=1,
        min_buy_quantity=1, adjustment_factor=1.0, signal_open=0.0,
        signal_high=0.0, signal_low=0.0, signal_close=0.0,
        factor_exposures={},
    )
    reference.enrich(snapshot)
    assert snapshot.close == 11.0
    assert snapshot.signal_close == 5.5
    assert snapshot.adjustment_factor == 0.5
    assert snapshot.industry == "Bank"
    assert snapshot.factor_exposures == {"value": 0.7}
    assert [action.timestamp for action in reference.actions_between(150, 300)] == [200, 300]


def test_arrow_batch_enrichment_matches_point_in_time_object_path():
    pa = pytest.importorskip("pyarrow")
    reference = MarketReferenceData(
        security_master=SecurityMaster([
            SecurityState(
                "X", 0, 200, industry="Bank", upper_limit=11.0,
                factor_exposures={"value": 0.7},
            ),
            SecurityState(
                "X", 200, None, is_st=True, industry="Tech",
                upper_limit=12.0, factor_exposures={"size": -0.2},
            ),
        ]),
        adjustment_factors=AdjustmentFactorStore([
            AdjustmentFactor("X", 0, 0.5, 200),
            AdjustmentFactor("X", 200, 2.0),
        ]),
    )
    batch = pa.record_batch({
        "timestamp": pa.array([100, 200], type=pa.int64()),
        "symbol": ["X", "X"],
        "open": [10.0, 10.0], "high": [11.0, 11.0],
        "low": [9.0, 9.0], "close": [10.0, 10.0],
        "volume": pa.array([100, 100], type=pa.int64()),
    })

    enriched = reference.enrich_batch(batch)

    assert enriched["industry"].to_pylist() == ["Bank", "Tech"]
    assert enriched["is_st"].to_pylist() == [False, True]
    assert enriched["adjustment_factor"].to_pylist() == [0.5, 2.0]
    assert enriched["signal_close"].to_pylist() == [5.0, 20.0]
    assert enriched["factor_exposure__value"].to_pylist() == [0.7, None]
    assert enriched["factor_exposure__size"].to_pylist() == [None, -0.2]
