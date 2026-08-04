from __future__ import annotations

import pytest

from dashboard.utils import equity_frame, page_window, rolling_sharpe


def test_dashboard_equity_uses_the_same_initial_cash_baseline():
    frame = equity_frame([
        {"date": "2026-01-02", "equity": 90.0, "cash": 50.0},
        {"date": "2026-01-03", "equity": 99.0, "cash": 60.0},
    ], initial_equity=100.0)

    assert frame["return"].tolist() == pytest.approx([-0.1, 0.1])
    assert frame["drawdown"].tolist() == pytest.approx([-0.1, -0.01])


def test_dashboard_page_window_is_bounded():
    assert page_window(251, 2, 100) == (2, 100, 100)
    assert page_window(251, 99, 100) == (3, 100, 200)


def test_dashboard_rolling_sharpe_subtracts_risk_free_return():
    import pandas as pd

    returns = pd.Series([0.001, 0.002, -0.001])
    actual = rolling_sharpe(returns, window=3, risk_free_rate=0.02).iloc[-1]
    expected = (
        (returns.mean() - 0.02 / 252.0)
        / returns.std(ddof=1)
        * (252.0 ** 0.5)
    )
    assert actual == pytest.approx(expected)
