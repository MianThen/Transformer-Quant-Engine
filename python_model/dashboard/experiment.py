"""本地数据导入与回测实验的业务服务。"""

from __future__ import annotations

import math
import os
import queue
import time as time_module
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from python.backtest_runner import BacktestRunner, BacktestResult
from python.data_feed import DataLakeFeed
from python.examples.ma_cross_strategy import MACrossStrategy
from python.examples.mean_reversion import MeanReversionStrategy
from python.examples.multi_symbol_strategies import (
    MultiSymbolMACrossStrategy,
    MultiSymbolMeanReversionStrategy,
)
from python.market_data import MinuteBarDataLake
from storage.trade_store import TradeStore


SHANGHAI = ZoneInfo("Asia/Shanghai")
SUPPORTED_SUFFIXES = {".csv", ".parquet", ".pq"}
UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024
EXPERIMENT_MODES = {"independent", "portfolio"}


@dataclass(frozen=True)
class ExperimentRequest:
    strategy: str
    symbol: str
    initial_cash: float
    order_size: int
    start: int | None = None
    end: int | None = None
    short_window: int = 5
    long_window: int = 20
    window: int = 20
    num_std: float = 2.0
    batch_size: int = 65_536
    mode: str = "independent"
    symbols: tuple[str, ...] = ()
    capital_utilization: float = 0.95

    def __post_init__(self) -> None:
        if self.strategy not in {"双均线", "均值回归"}:
            raise ValueError("不支持的策略")
        normalized = tuple(dict.fromkeys(
            str(value).strip() for value in (self.symbols or (self.symbol,))
        ))
        if not normalized or any(not value for value in normalized):
            raise ValueError("symbols 不能为空")
        object.__setattr__(self, "symbols", normalized)
        object.__setattr__(self, "symbol", normalized[0])
        if self.mode not in EXPERIMENT_MODES:
            raise ValueError("不支持的回测模式")
        if not math.isfinite(self.initial_cash) or self.initial_cash < 0.0:
            raise ValueError("initial_cash 必须是有限非负数")
        if isinstance(self.order_size, bool) or self.order_size <= 0:
            raise ValueError("order_size 必须为正整数")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("开始时间不能晚于结束时间")
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须为正数")
        if (
            not math.isfinite(self.capital_utilization)
            or not 0.0 < self.capital_utilization < 1.0
        ):
            raise ValueError("capital_utilization 必须在 (0, 1) 内")
        if self.strategy == "双均线" and not (
            1 <= self.short_window < self.long_window
        ):
            raise ValueError("双均线参数必须满足 1 <= short_window < long_window")
        if self.strategy == "均值回归" and (
            self.window < 2 or not math.isfinite(self.num_std) or self.num_std <= 0.0
        ):
            raise ValueError("均值回归窗口至少为 2，标准差倍数必须为正数")

    def strategy_parameters(self) -> dict[str, object]:
        if self.mode == "portfolio":
            common = {
                "symbols": list(self.symbols),
                "capital_utilization": self.capital_utilization,
                "lot_size": 100,
            }
            if self.strategy == "双均线":
                return {
                    **common,
                    "short_window": self.short_window,
                    "long_window": self.long_window,
                }
            return {**common, "window": self.window, "num_std": self.num_std}
        common = {"symbol": self.symbol, "order_size": self.order_size}
        if self.strategy == "双均线":
            return {
                **common,
                "short_window": self.short_window,
                "long_window": self.long_window,
            }
        return {**common, "window": self.window, "num_std": self.num_std}


@dataclass(frozen=True)
class IndependentExperimentResult:
    symbol: str
    result: BacktestResult | None = None
    error: str = ""


@dataclass(frozen=True)
class IndependentExperimentProgress:
    completed: int
    total: int
    symbol: str


def run_experiment(
    lake: MinuteBarDataLake,
    database: str | Path,
    request: ExperimentRequest,
) -> BacktestResult:
    if request.mode == "independent" and len(request.symbols) != 1:
        raise ValueError("多标的独立回测请使用 run_independent_experiments")
    parameters = request.strategy_parameters()
    if request.mode == "portfolio" and request.strategy == "双均线":
        strategy = MultiSymbolMACrossStrategy(
            initial_cash=request.initial_cash, **parameters
        )
    elif request.mode == "portfolio":
        strategy = MultiSymbolMeanReversionStrategy(
            initial_cash=request.initial_cash, **parameters
        )
    elif request.strategy == "双均线":
        strategy = MACrossStrategy(**parameters)
    else:
        strategy = MeanReversionStrategy(**parameters)
    feed = DataLakeFeed(
        lake,
        symbols=request.symbols,
        start=request.start,
        end=request.end,
        batch_size=request.batch_size,
    )
    store = TradeStore(database)
    try:
        return BacktestRunner(
            strategy,
            feed,
            initial_cash=request.initial_cash,
            store=store,
            strategy_parameters=parameters,
        ).run()
    finally:
        store.close()


def run_independent_experiments(
    lake: MinuteBarDataLake,
    database: str | Path,
    request: ExperimentRequest,
    progress=None,
) -> tuple[IndependentExperimentResult, ...]:
    if request.mode != "independent":
        raise ValueError("独立批量回测需要 independent 模式")
    outcomes = []
    total = len(request.symbols)
    for completed, symbol in enumerate(request.symbols, start=1):
        single = replace(request, symbol=symbol, symbols=(symbol,))
        try:
            result = run_experiment(lake, database, single)
            outcomes.append(IndependentExperimentResult(symbol, result=result))
        except Exception as error:
            outcomes.append(IndependentExperimentResult(
                symbol, error=f"{type(error).__name__}: {error}"
            ))
        if progress is not None:
            progress(IndependentExperimentProgress(completed, total, symbol))
    return tuple(outcomes)


def zero_trade_diagnostic(
    result, *, initial_cash: float, order_size: int | None,
) -> str | None:
    """Explain a completed run with no fills using its order audit."""
    if result.trades:
        return None
    if not result.orders:
        return "回测完成但没有成交：策略在所选区间内没有产生委托。"

    reasons = Counter(
        getattr(record.reject_reason, "name", str(record.reject_reason).rsplit(".", 1)[-1])
        for record in result.orders
        if getattr(record.reject_reason, "name", None) != "NONE"
    )
    cash_rejections = reasons.get("INSUFFICIENT_CASH", 0)
    if cash_rejections:
        if order_size is None:
            return (
                f"回测完成但没有成交：{cash_rejections} 笔委托因组合可用资金不足被拒绝。"
                "请降低组合资金利用率或增加本金。"
            )
        affordable = initial_cash / order_size if order_size > 0 else 0.0
        return (
            f"回测完成但没有成交：{cash_rejections} 笔委托因资金不足被拒绝。"
            f"本金 {initial_cash:,.2f} 元、目标 {order_size:,} 股，"
            f"每股可用资金约 {affordable:,.2f} 元（未计手续费）。"
            "请降低目标数量或增加本金。"
        )
    if reasons:
        labels = {
            "INVALID_LOT_SIZE": "数量不符合整手规则",
            "INSUFFICIENT_POSITION": "可卖持仓不足",
            "NOT_LISTED": "标的未上市",
            "STALE_MARKET_DATA": "行情数据过期",
            "UNKNOWN_SYMBOL": "未知标的",
            "INVALID_ORDER": "委托参数无效",
        }
        details = "、".join(
            f"{labels.get(reason, reason)} {count} 笔"
            for reason, count in reasons.most_common()
        )
        return f"回测完成但没有成交：{details}。"
    return "回测完成但没有成交：委托未最终成交，请在运行审计中查看详情。"


def ingest_with_progress(
    lake: MinuteBarDataLake,
    source: str | Path,
    progress=None,
    *,
    poll_interval: float = 0.05,
):
    """Run Arrow ingestion off the UI thread and report progress on the caller thread."""
    if poll_interval <= 0:
        raise ValueError("poll_interval 必须为正数")
    return run_with_progress(
        lambda report: lake.ingest(source, progress=report),
        progress,
        poll_interval=poll_interval,
    )


def ingest_directory_with_progress(
    lake: MinuteBarDataLake,
    source: str | Path,
    progress=None,
    *,
    file_batch_size: int = 10_000,
    poll_interval: float = 0.05,
):
    root = Path(source).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    total_files = count_source_files(root)
    if total_files == 0:
        raise ValueError(f"目录中没有 CSV/Parquet 文件: {root}")
    return run_with_progress(
        lambda report: lake.ingest_many(
            discover_source_files(root),
            progress=report,
            file_batch_size=file_batch_size,
            total_files=total_files,
        ),
        progress,
        poll_interval=poll_interval,
    )


def run_with_progress(operation, progress=None, *, poll_interval: float = 0.05):
    if poll_interval <= 0:
        raise ValueError("poll_interval 必须为正数")
    updates: queue.SimpleQueue = queue.SimpleQueue()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="qbt-ingest") as pool:
        future = pool.submit(operation, updates.put)
        while not future.done():
            _drain_progress(updates, progress)
            time_module.sleep(poll_interval)
        _drain_progress(updates, progress)
        return future.result()


def discover_source_files(root: str | Path):
    """Yield source files deterministically without materialising the full path list."""
    pending = [Path(root).expanduser().resolve()]
    while pending:
        current = pending.pop()
        directories = []
        files = []
        with os.scandir(current) as entries:
            for entry in entries:
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    directories.append(path)
                elif (
                    entry.is_file(follow_symlinks=False)
                    and path.suffix.lower() in SUPPORTED_SUFFIXES
                ):
                    files.append(path)
        for path in sorted(files):
            yield path
        pending.extend(reversed(sorted(directories)))


def count_source_files(root: str | Path) -> int:
    return sum(1 for _ in discover_source_files(root))


def _drain_progress(updates, progress) -> None:
    while True:
        try:
            value = updates.get_nowait()
        except queue.Empty:
            return
        if progress is not None:
            progress(value)


def validate_source_path(value: str | Path, *, allow_directory: bool = False) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        if allow_directory:
            return path
        raise IsADirectoryError(f"请输入单个 CSV/Parquet 文件，目录请使用批量导入: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError("仅支持 CSV、Parquet 和 PQ 文件")
    return path


def save_uploaded_file(uploaded, lake_root: str | Path) -> Path:
    name = Path(str(uploaded.name)).name
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("仅支持 CSV、Parquet 和 PQ 文件")
    destination_dir = Path(lake_root).expanduser().resolve() / "uploads"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{uuid.uuid4().hex}-{name}"
    uploaded.seek(0)
    with destination.open("xb") as file:
        while True:
            chunk = uploaded.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            file.write(chunk)
    return destination


def local_datetime_to_ns(day: date, clock: time) -> int:
    value = datetime.combine(day, clock, tzinfo=SHANGHAI)
    return int(value.timestamp() * 1_000_000_000)


def ns_to_local_datetime(timestamp: int) -> datetime:
    return datetime.fromtimestamp(timestamp / 1_000_000_000, tz=SHANGHAI)


def format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def describe_exception(error: BaseException) -> str:
    """Return useful UI text even for exceptions with an empty message."""
    message = str(error).strip()
    return f"{type(error).__name__}: {message or repr(error)}"
