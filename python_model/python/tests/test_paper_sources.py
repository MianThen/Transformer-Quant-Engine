"""虚拟盘数据层回归测试:上游边界探测与 warmup 扩窗。

背景:backfill 原用固定 [-60d, now] 探针判断 scrapy 边界,scrapy 停更超过
60 天后探针落空,回补整体中止("探测不到近期行情")。以下用例锁定修复后的
行为:边界必须等于数据真实末尾,且探测失败要逐级放大而不是直接放弃。
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python.paper import runner, sources  # noqa: E402

_NS_PER_DAY = sources.NS_PER_DAY
# 冻结"现在",避免 scrapy_boundary 内部取 now() 与断言取值产生时钟漂移
_FROZEN = dt.datetime(2026, 9, 21, 7, 0, tzinfo=dt.timezone.utc)
_FROZEN_NS = int(_FROZEN.timestamp()) * 1_000_000_000


class _FrozenDatetime(dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return _FROZEN if tz is not None else _FROZEN.replace(tzinfo=None)


@pytest.fixture(autouse=True)
def _freeze_now(monkeypatch):
    monkeypatch.setattr(sources.dt, "datetime", _FrozenDatetime)


def _frame(days_ago: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {"timestamp": [_FROZEN_NS - n * _NS_PER_DAY for n in days_ago]}
    )


def test_scrapy_boundary_returns_latest_ts(monkeypatch):
    monkeypatch.setattr(
        sources, "load_scrapy_frame", lambda *a, **k: _frame([3, 10, 40])
    )
    assert sources.scrapy_boundary(["000001"]) == _FROZEN_NS - 3 * _NS_PER_DAY


def test_scrapy_boundary_widens_window_when_recent_is_empty(monkeypatch):
    """近期窗口空 -> 必须继续向前探测,而不是返回 None 让回补整体中止。"""
    calls: list[int] = []
    stale_ns = _FROZEN_NS - 300 * _NS_PER_DAY

    def fake_load(symbols, start_ns, end_ns):
        calls.append(start_ns)
        # 只有窗口放宽到 ~120 天以上才看得到数据(模拟 scrapy 停更 300 天)
        return _frame([300]) if (end_ns - start_ns) > 120 * _NS_PER_DAY else pd.DataFrame()

    monkeypatch.setattr(sources, "load_scrapy_frame", fake_load)
    assert sources.scrapy_boundary(["000001"]) == stale_ns
    assert len(calls) >= 2, "应当逐级放大探测窗口"


def test_scrapy_boundary_returns_none_when_data_absent(monkeypatch):
    monkeypatch.setattr(sources, "load_scrapy_frame", lambda *a, **k: pd.DataFrame())
    assert sources.scrapy_boundary(["000001"]) is None


class _StubConnection:
    def __init__(self, watermark):
        self._watermark = watermark

    def execute(self, _sql, _params=()):
        return _StubCursor(self._watermark)


class _StubCursor:
    def __init__(self, watermark):
        self._watermark = watermark

    def fetchone(self):
        return (self._watermark,)


def test_source_boundary_prefers_later_of_scrapy_and_staging(monkeypatch):
    scrapy_ns = 1_000 * _NS_PER_DAY
    monkeypatch.setattr(sources, "scrapy_boundary", lambda symbols: scrapy_ns)

    later, basis = sources.source_boundary_ns(["000001"], _StubConnection(scrapy_ns + 5))
    assert (later, basis) == (scrapy_ns + 5, "staging")

    earlier, basis = sources.source_boundary_ns(["000001"], _StubConnection(scrapy_ns - 5))
    assert (earlier, basis) == (scrapy_ns, "scrapy")


def test_source_boundary_none_only_when_both_sources_empty(monkeypatch):
    monkeypatch.setattr(sources, "scrapy_boundary", lambda symbols: None)
    assert sources.source_boundary_ns(["000001"], _StubConnection(None)) == (None, "无")

    staging_only, basis = sources.source_boundary_ns(["000001"], _StubConnection(7))
    assert (staging_only, basis) == (7, "staging")


@pytest.mark.parametrize(
    "warmup_days,expected",
    [(170, 420), (0, 420), (1000, 1780)],
)
def test_warmup_probe_days_keeps_backward_compatible_floor(warmup_days, expected):
    assert runner._warmup_probe_days(warmup_days) == expected
