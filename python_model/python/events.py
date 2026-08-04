"""历史回放与未来实时源共用的批量事件契约。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, Iterator
from zoneinfo import ZoneInfo

from .market_data.reference import FACTOR_EXPOSURE_COLUMN_PREFIX


SHANGHAI = ZoneInfo("Asia/Shanghai")


class EventType(str, Enum):
    MARKET_DATA_BATCH = "market_data_batch"
    CORPORATE_ACTION = "corporate_action"
    TIMER = "timer"
    SESSION = "session"


@dataclass(frozen=True)
class MarketDataBatchEvent:
    timestamp: int
    record_batch: object
    type: EventType = EventType.MARKET_DATA_BATCH


@dataclass(frozen=True)
class CorporateActionEvent:
    timestamp: int
    action: object
    type: EventType = EventType.CORPORATE_ACTION


@dataclass(frozen=True)
class TimerEvent:
    timestamp: int
    name: str
    type: EventType = EventType.TIMER


@dataclass(frozen=True)
class SessionEvent:
    timestamp: int
    state: str
    trading_date: str
    type: EventType = EventType.SESSION


class MarketEventSource(ABC):
    @abstractmethod
    def events(self) -> Iterator[object]:
        raise NotImplementedError

    def __iter__(self):
        return self.events()


class LiveMarketSource(MarketEventSource):
    """未来实时适配器接口；网络连接和重连策略不属于 P2。"""


class HistoricalReplaySource(MarketEventSource):
    def __init__(
        self,
        replay,
        *,
        symbols=None,
        start=None,
        end=None,
        columns=("timestamp", "symbol", "open", "high", "low", "close", "volume"),
        calendar=None,
        reference_data=None,
        timers: Iterable[tuple[int, str]] = (),
    ) -> None:
        self.replay = replay
        self.query = {
            "symbols": symbols, "start": start, "end": end, "columns": columns,
        }
        self.calendar = calendar
        self.reference_data = reference_data
        self.timers = tuple(sorted((int(ts), str(name)) for ts, name in timers))

    def lineage(self):
        if hasattr(self.replay, "stream_batches"):
            return self.replay.lineage()
        return self.replay.lineage(**self.query)

    def events(self):
        timer_index = 0
        previous_timestamp = None
        previous_date = None
        batches = (
            self.replay.stream_batches()
            if hasattr(self.replay, "stream_batches")
            else self.replay.iter_batches(**self.query)
        )
        for batch in _timestamp_atomic_batches(batches):
            if self.reference_data is not None:
                batch = self.reference_data.enrich_batch(batch)
            timestamp_index = batch.schema.get_field_index("timestamp")
            timestamps = batch.column(timestamp_index)
            offset = 0
            segment_start = 0
            while offset < batch.num_rows:
                timestamp = int(timestamps[offset].as_py())
                boundary = offset + 1
                while boundary < batch.num_rows and timestamps[boundary].as_py() == timestamp:
                    boundary += 1
                sparse_events = []
                if timestamp != previous_timestamp:
                    if self.calendar is not None:
                        self.calendar.validate_timestamp(timestamp)
                    trading_date = _trading_date(timestamp)
                    if trading_date != previous_date:
                        if previous_date is not None:
                            sparse_events.append(
                                SessionEvent(previous_timestamp, "CLOSE", previous_date)
                            )
                        sparse_events.append(SessionEvent(timestamp, "OPEN", trading_date))
                        previous_date = trading_date

                    if self.reference_data is not None:
                        actions = (
                            self.reference_data.actions_at(timestamp)
                            if previous_timestamp is None
                            else self.reference_data.actions_between(
                                previous_timestamp, timestamp
                            )
                        )
                        sparse_events.extend(
                            CorporateActionEvent(action.timestamp, action)
                            for action in actions
                        )

                    while (timer_index < len(self.timers)
                           and self.timers[timer_index][0] <= timestamp):
                        timer_timestamp, name = self.timers[timer_index]
                        if previous_timestamp is None or timer_timestamp > previous_timestamp:
                            sparse_events.append(TimerEvent(timer_timestamp, name))
                        timer_index += 1

                if sparse_events:
                    if offset > segment_start:
                        first_timestamp = int(timestamps[segment_start].as_py())
                        yield MarketDataBatchEvent(
                            first_timestamp,
                            batch.slice(segment_start, offset - segment_start),
                        )
                    yield from sparse_events
                    segment_start = offset
                previous_timestamp = timestamp
                offset = boundary
            if segment_start < batch.num_rows:
                first_timestamp = int(timestamps[segment_start].as_py())
                yield MarketDataBatchEvent(
                    first_timestamp, batch.slice(segment_start)
                )
        if previous_date is not None:
            yield SessionEvent(previous_timestamp, "CLOSE", previous_date)


@dataclass(frozen=True)
class EventRunStats:
    events: int
    market_batches: int
    rows: int
    stream_calls: int = 0
    last_timestamp: int = 0


class BatchEventRunner:
    """消费统一事件接口；策略回调仍由 BacktestEngine 管理。"""

    def __init__(
        self, engine, *, on_timer=None, on_session=None, finalize: bool = True
    ) -> None:
        self.engine = engine
        self.on_timer = on_timer
        self.on_session = on_session
        self.should_finalize = finalize

    def run(self, source: MarketEventSource) -> EventRunStats:
        events = market_batches = rows = stream_calls = 0
        last_timestamp = 0
        iterator = iter(source.events())
        pending = None
        while True:
            if pending is not None:
                event = pending
                pending = None
            else:
                try:
                    event = next(iterator)
                except StopIteration:
                    break
            events += 1
            last_timestamp = max(last_timestamp, int(event.timestamp))
            if event.type == EventType.MARKET_DATA_BATCH:
                if hasattr(self.engine, "process_arrow_stream"):
                    deferred = []

                    def continuous_batches():
                        nonlocal events, market_batches, rows, last_timestamp
                        current = event
                        while True:
                            batch = current.record_batch
                            market_batches += 1
                            rows += batch.num_rows
                            last_timestamp = max(
                                last_timestamp, _last_batch_timestamp(batch)
                            )
                            yield batch
                            try:
                                current = next(iterator)
                            except StopIteration:
                                return
                            if current.type != EventType.MARKET_DATA_BATCH:
                                deferred.append(current)
                                return
                            events += 1

                    self._process_market_stream(
                        event.record_batch.schema, continuous_batches()
                    )
                    stream_calls += 1
                    if deferred:
                        pending = deferred[0]
                else:
                    self._process_market_batch(event.record_batch)
                    market_batches += 1
                    rows += event.record_batch.num_rows
                    last_timestamp = max(
                        last_timestamp, _last_batch_timestamp(event.record_batch)
                    )
            elif event.type == EventType.CORPORATE_ACTION:
                self._apply_action(event.action)
            elif event.type == EventType.TIMER and self.on_timer is not None:
                self.on_timer(event)
            elif event.type == EventType.SESSION and self.on_session is not None:
                self.on_session(event)
        if self.should_finalize and hasattr(self.engine, "finalize"):
            self.engine.finalize(last_timestamp)
        return EventRunStats(
            events, market_batches, rows, stream_calls, last_timestamp
        )

    def _process_market_stream(self, schema, batches) -> None:
        import pyarrow as pa

        reader = pa.RecordBatchReader.from_batches(schema, batches)
        self.engine.process_arrow_stream(reader)

    def _process_market_batch(self, batch) -> None:
        import pyarrow as pa

        if hasattr(self.engine, "process_arrow_stream"):
            reader = pa.RecordBatchReader.from_batches(batch.schema, [batch])
            self.engine.process_arrow_stream(reader)
            return
        from .engine_api import MarketSnapshot

        values = {name: batch[name].to_pylist() for name in batch.schema.names}
        snapshots = []
        for index in range(batch.num_rows):
            snapshot = MarketSnapshot()
            for name in ("timestamp", "symbol", "open", "high", "low", "close", "volume"):
                setattr(snapshot, name, values[name][index])
            for name in (
                "upper_limit", "lower_limit", "is_suspended", "is_listed", "is_st",
                "lot_size", "min_buy_quantity", "board", "industry",
                "adjustment_factor", "signal_open", "signal_high", "signal_low",
                "signal_close",
            ):
                if name in values and values[name][index] is not None:
                    setattr(snapshot, name, values[name][index])
            snapshot.factor_exposures = {
                name[len(FACTOR_EXPOSURE_COLUMN_PREFIX):]: column[index]
                for name, column in values.items()
                if name.startswith(FACTOR_EXPOSURE_COLUMN_PREFIX)
                and column[index] is not None
            }
            snapshots.append(snapshot)
        start = 0
        while start < len(snapshots):
            end = start + 1
            while (end < len(snapshots)
                   and snapshots[end].timestamp == snapshots[start].timestamp):
                end += 1
            self.engine.process_market_data_batch(snapshots[start:end])
            start = end

    def _apply_action(self, source) -> None:
        from .engine_api import CorporateAction

        action = CorporateAction()
        for name in (
            "symbol", "timestamp", "cash_dividend_per_share",
            "share_multiplier", "description",
        ):
            setattr(action, name, getattr(source, name))
        self.engine.apply_corporate_action(action)


def _trading_date(timestamp: int) -> str:
    return datetime.fromtimestamp(
        timestamp / 1e9, tz=timezone.utc
    ).astimezone(SHANGHAI).date().isoformat()


def _last_batch_timestamp(batch) -> int:
    if batch.num_rows == 0:
        return 0
    return int(batch["timestamp"][batch.num_rows - 1].as_py())


def _timestamp_atomic_batches(batches):
    """合并跨 RecordBatch 的同一 timestamp，最多暂存一个完整截面。"""
    schema = None
    previous_last_key = None
    carry = []
    carry_timestamp = None

    for source_batch in batches:
        if schema is None:
            schema = source_batch.schema
        elif not source_batch.schema.equals(schema):
            raise ValueError("market data RecordBatch schema changed during replay")
        if source_batch.num_rows == 0:
            continue

        first_key, last_key = _validate_ordered_batch(source_batch)
        if previous_last_key is not None and first_key <= previous_last_key:
            raise ValueError(
                "market data batches must be strictly ordered by "
                "(timestamp, symbol)"
            )
        previous_last_key = last_key

        batch = source_batch
        completed = []
        if carry:
            if first_key[0] == carry_timestamp:
                boundary = _timestamp_upper_bound(
                    batch["timestamp"], carry_timestamp
                )
                carry.append(batch.slice(0, boundary))
                batch = batch.slice(boundary)
                if batch.num_rows == 0:
                    continue
            completed.extend(carry)
            carry = []

        timestamps = batch["timestamp"]
        carry_timestamp = int(timestamps[batch.num_rows - 1].as_py())
        tail_start = _timestamp_lower_bound(timestamps, carry_timestamp)
        if tail_start > 0:
            completed.append(batch.slice(0, tail_start))
        carry = [batch.slice(tail_start)]

        if completed:
            yield _combine_record_batches(completed)

    if carry:
        yield _combine_record_batches(carry)


def _validate_ordered_batch(batch):
    import pyarrow.compute as pc

    required = ("timestamp", "symbol")
    missing = [name for name in required if name not in batch.schema.names]
    if missing:
        raise ValueError(
            "market data RecordBatch missing columns: " + ", ".join(missing)
        )
    timestamps = batch["timestamp"]
    symbols = batch["symbol"]
    if timestamps.null_count or symbols.null_count:
        raise ValueError("market data timestamp and symbol cannot be null")
    if batch.num_rows > 1:
        earlier_timestamps = timestamps.slice(0, batch.num_rows - 1)
        later_timestamps = timestamps.slice(1)
        earlier_symbols = symbols.slice(0, batch.num_rows - 1)
        later_symbols = symbols.slice(1)
        ordered = pc.or_(
            pc.greater(later_timestamps, earlier_timestamps),
            pc.and_(
                pc.equal(later_timestamps, earlier_timestamps),
                pc.greater(later_symbols, earlier_symbols),
            ),
        )
        if not pc.all(ordered).as_py():
            raise ValueError(
                "market data batches must be strictly ordered by "
                "(timestamp, symbol)"
            )
    return (
        (int(timestamps[0].as_py()), str(symbols[0].as_py())),
        (
            int(timestamps[batch.num_rows - 1].as_py()),
            str(symbols[batch.num_rows - 1].as_py()),
        ),
    )


def _timestamp_lower_bound(timestamps, value: int) -> int:
    left = 0
    right = len(timestamps)
    while left < right:
        middle = (left + right) // 2
        if int(timestamps[middle].as_py()) < value:
            left = middle + 1
        else:
            right = middle
    return left


def _timestamp_upper_bound(timestamps, value: int) -> int:
    left = 0
    right = len(timestamps)
    while left < right:
        middle = (left + right) // 2
        if int(timestamps[middle].as_py()) <= value:
            left = middle + 1
        else:
            right = middle
    return left


def _combine_record_batches(batches):
    if len(batches) == 1:
        return batches[0]
    import pyarrow as pa

    table = pa.Table.from_batches(batches).combine_chunks()
    combined = table.to_batches(max_chunksize=table.num_rows)
    if len(combined) != 1:
        raise RuntimeError("failed to combine timestamp-atomic RecordBatch")
    return combined[0]
