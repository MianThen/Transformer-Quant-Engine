"""虚拟盘会话配置与路径常量。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# scrapy 下载器(行情上游)。口径不一致时必须换成本项目的路径,不要跨机复制 paper.db
SCRAPY_PROJECT_ROOT = Path(
    os.environ.get("QBT_SCRAPY_ROOT", "/Users/Zhuanz/PycharmProjects/scrapy")
)

# 行情上游(与 phase1e 同源,见 configs/ml/phase1e_diagnostics.local.json)
SCRAPY_BARS_ROOT = Path(
    "/Users/Zhuanz/PycharmProjects/scrapy/data/bars/"
    "provider=baostock/frequency=1d/adjustment=none"
)
SCRAPY_STATE_ROOT = Path(
    "/Users/Zhuanz/PycharmProjects/scrapy/data/security_state/provider=baostock"
)

# 上游下载器自己的虚拟环境(其依赖独立于本项目)
SCRAPY_PYTHON = Path(os.environ.get(
    "QBT_SCRAPY_PYTHON", str(SCRAPY_PROJECT_ROOT / ".venv/bin/python")
))
SCRAPY_BARS_ENTRY = SCRAPY_PROJECT_ROOT / "main.py"
SCRAPY_STATE_ENTRY = SCRAPY_PROJECT_ROOT / "download_baostock_state.py"

# 冠军制品(sha 0ef338fe…,三方一致性登记 PASS)
ARTIFACT_PATH = PROJECT_ROOT / "models" / "champion-legacy-fold3-safe"

# 运行时产物目录(paper.db、宇宙缓存、日志)
# 可用 QBT_PAPER_DIR 覆盖(沙箱 / 只读工作区分册运行时产物)
PAPER_DIR = Path(os.environ.get(
    "QBT_PAPER_DIR", str(PROJECT_ROOT / "runs" / "paper")
))
DB_PATH = PAPER_DIR / "paper.db"
UNIVERSE_CACHE = PAPER_DIR / "universe_pit120.json"

# pybind11 ML 构建的 cpp_engine 模块目录(可用环境变量覆盖)
ENGINE_MODULE_DIR = Path(os.environ.get(
    "QBT_CPP_ENGINE_PATH",
    "/Users/Zhuanz/CLionProjects/quant-backtester-cpp/build/ml/cpp_engine",
))

DEFAULT_INITIAL_CASH = 1_000_000.0
DEFAULT_START_DATE = "2026-01-02"  # 虚拟盘首个交易日(回放演练起点)

# BAR_V1 需 64 lookback + 60 特征窗 ≈ 124 根 bar,留裕量
WARMUP_TRADING_DAYS = 170

# pit-120 宇宙预注册选择(见 data/research/phase1e_pit_120_audit.json)
UNIVERSE_AUDIT_SHA256 = (
    "5355ef48fee839cc0aee1fe9bced3add10a9719247ba05731084e52c26414144"
)
UNIVERSE_SIZE = 120


@dataclass(frozen=True)
class SessionSpec:
    """一种虚拟盘会话预设。"""

    kind: str
    session_id: str
    policy: dict
    note: str
    cost_profile: str | None = None  # None = 历史零成本口径


# registered: 登记口径原样(minimum_confidence=0.55,9f4ca0e 零订单验证同款配置)
# observed: 放宽 confidence 门槛让 top-20 排名实际成交,用于观察模型真实选股行为
# observed-cost: 与 observed 同策略、同行情,唯一变量是显性交易成本,
#                用于回答"毛收益扣掉真实费用后还剩多少"
SESSION_PRESETS: tuple[SessionSpec, ...] = (
    SessionSpec(
        kind="registered", session_id="registered",
        policy={"minimum_confidence": 0.55},
        note="登记口径原样;预期订单稀疏/为零,忠实反映",
    ),
    SessionSpec(
        kind="observed", session_id="observed",
        policy={"minimum_confidence": 0.0},
        note="观测对照:放宽 confidence,展示 top-20 排名实际交易行为(零成本基线)",
    ),
    SessionSpec(
        kind="observed", session_id="observed-cost",
        policy={"minimum_confidence": 0.0},
        note="成本对照:同 observed 策略与行情,登记佣金/印花税/过户费(无滑点)",
        cost_profile="ashare-w1",
    ),
    # 滑点档:低价股半档价差量级(实测 tick/价格中位 13.9bp),用于给出成本上界
    SessionSpec(
        kind="comparison", session_id="observed-cost-spread",
        policy={"minimum_confidence": 0.0},
        note="成本上界:显性费用 + 每边让 1 个最小报价单位(≈14bp 滑点)",
        cost_profile="ashare-w1-spread",
    ),
)


def session_spec(session_id: str) -> SessionSpec:
    for spec in SESSION_PRESETS:
        if spec.session_id == session_id:
            return spec
    known = ", ".join(s.session_id for s in SESSION_PRESETS)
    raise ValueError(f"未知会话 {session_id};可用: {known}")
