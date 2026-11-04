"""pit-120 宇宙清单:按 phase1e 预注册选择规则重算并与审计 sha 对拍。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import (
    SCRAPY_BARS_ROOT,
    UNIVERSE_AUDIT_SHA256,
    UNIVERSE_CACHE,
    UNIVERSE_SIZE,
)


def _canonical_hash(symbols: list[str]) -> str:
    payload = json.dumps(
        symbols, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_universe() -> list[str]:
    """返回 120 标的清单;优先用缓存,首次从 scrapy 2019 分区重算并校验审计 sha。"""
    if UNIVERSE_CACHE.exists():
        symbols = json.loads(UNIVERSE_CACHE.read_text(encoding="utf-8"))
        if len(symbols) != UNIVERSE_SIZE:
            raise RuntimeError(f"宇宙缓存损坏:{UNIVERSE_CACHE}")
        return symbols

    import duckdb

    root = str(SCRAPY_BARS_ROOT.resolve())
    if "'" in root:
        raise ValueError("数据路径不能包含单引号")
    connection = duckdb.connect()
    connection.execute(f"""
        CREATE TEMP TABLE selected AS
        SELECT symbol
        FROM read_parquet('{root}/ticker=*/year=2019/*.parquet',
                          hive_partitioning=true)
        GROUP BY symbol
        HAVING count(DISTINCT timestamp) >= 220
        ORDER BY symbol
        LIMIT {UNIVERSE_SIZE}
    """)
    symbols = [
        row[0] for row in connection.execute(
            "SELECT symbol FROM selected ORDER BY symbol"
        ).fetchall()
    ]
    if len(symbols) != UNIVERSE_SIZE:
        raise RuntimeError("满足预注册历史覆盖条件的股票不足,宇宙无法重建")

    digest = _canonical_hash(symbols)
    if digest != UNIVERSE_AUDIT_SHA256:
        raise RuntimeError(
            f"宇宙清单 sha256 不匹配:重算 {digest} != 审计 {UNIVERSE_AUDIT_SHA256}"
        )

    UNIVERSE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    UNIVERSE_CACHE.write_text(
        json.dumps(symbols, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return symbols


def verify_universe(symbols: list[str]) -> None:
    digest = _canonical_hash(symbols)
    if digest != UNIVERSE_AUDIT_SHA256:
        raise RuntimeError(f"宇宙清单 sha256 校验失败:{digest}")
