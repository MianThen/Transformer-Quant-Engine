"""日频虚拟盘(paper trading)harness。

冠军制品(champion-legacy-fold3-safe)的观测环境:replay-to-date 逐日推进,
C++ 引擎(pybind11 cpp_engine, ML 构建)负责撮合与账务,结果落 runs/paper/paper.db。

纯虚拟盘:不连接任何真实下单通道;制品 promotion_eligible=false,
本包产出只作行为观测,不构成晋级证据(见 docs/champion_artifact_registration.md)。
"""
