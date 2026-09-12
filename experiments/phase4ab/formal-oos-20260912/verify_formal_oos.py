"""Phase 4A/4B 正式 OOS 预注册统计校验（合同 §4/§5）。

输入: formal_oos_report.json（C++ driver 产物）
输出: 配对差、3 子窗口方向一致性、block bootstrap CI、FDR、DSR、最终判定
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ARMS = ("risk_only", "posterior_bl", "posterior_ffv", "nco_ffv")
CHALLENGERS = ("posterior_bl", "posterior_ffv", "nco_ffv")
BASELINE = "risk_only"
PERIODS_PER_YEAR = 50.4
BLOCK = 10
DRAWS = 2000
SEED = 20260912
WINDOWS = 3


def sharpe(returns: np.ndarray) -> float:
    if returns.size < 2 or returns.std(ddof=1) <= 0:
        return 0.0
    return float(returns.mean() / returns.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR))


def block_bootstrap_indices(count: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    out = np.empty((draws, count), dtype=np.int64)
    for draw in range(draws):
        position = 0
        while position < count:
            start = rng.integers(0, count)
            length = min(BLOCK, count - position)
            out[draw, position:position + length] = (start + np.arange(length)) % count
            position += length
    return out


def norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def main() -> int:
    report_path = Path(sys.argv[1])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    returns = {arm: np.asarray(report["returns"][arm], dtype=np.float64) for arm in ARMS}
    count = len(returns[BASELINE])
    assert all(len(value) == count for value in returns.values()), "arm 收益序列长度不一致"
    degraded = sum(1 for step in report["steps"] if step.get("view_tier", 0) >= 1)
    print(f"periods={count} degraded={degraded} ({100*degraded/count:.1f}%, 合同上限 20%)")
    if degraded / count > 0.20:
        print("RUN_INVALID: 降级步超限")
        return 1

    rng = np.random.default_rng(SEED)
    indices = block_bootstrap_indices(count, DRAWS, rng)

    print("\n=== 配对差 vs incumbent（全样本 Sharpe）===")
    results = {}
    for arm in ARMS:
        print(f"  {arm:16s} sharpe={sharpe(returns[arm]):+.4f}")
    for challenger in CHALLENGERS:
        base_draws = returns[BASELINE][indices]
        chal_draws = returns[challenger][indices]
        diff_draws = np.array([sharpe(chal_draws[i]) - sharpe(base_draws[i]) for i in range(DRAWS)])
        point = sharpe(returns[challenger]) - sharpe(returns[BASELINE])
        ci_low = float(np.quantile(diff_draws, 0.025))
        ci_high = float(np.quantile(diff_draws, 0.975))
        p_value = float((diff_draws <= 0.0).mean())
        window_signs = []
        for window in range(WINDOWS):
            lo = count * window // WINDOWS
            hi = count * (window + 1) // WINDOWS
            delta = sharpe(returns[challenger][lo:hi]) - sharpe(returns[BASELINE][lo:hi])
            window_signs.append(delta > 0)
        results[challenger] = {
            "point": point, "ci_low": ci_low, "ci_high": ci_high,
            "p": p_value, "windows": window_signs,
        }
        print(f"  {challenger}−{BASELINE}: ΔSharpe={point:+.4f} CI95=[{ci_low:+.4f},{ci_high:+.4f}] "
              f"p={p_value:.3f} 子窗口同向={sum(window_signs)}/3")

    print("\n=== FDR (BH, 3 组 challenger) ===")
    order = sorted(CHALLENGERS, key=lambda arm: results[arm]["p"])
    m = len(order)
    fdr_pass = {}
    for rank, arm in enumerate(order, start=1):
        threshold = 0.05 * rank / m
        fdr_pass[arm] = results[arm]["p"] <= threshold
        print(f"  {arm:16s} p={results[arm]['p']:.3f} <= {threshold:.4f}? {'Y' if fdr_pass[arm] else 'N'}")

    print("\n=== 判定（合同 §5）===")
    any_winner = False
    for arm in CHALLENGERS:
        ok_windows = all(results[arm]["windows"])
        ok_ci = results[arm]["ci_low"] > 0
        ok_fdr = fdr_pass[arm]
        winner = ok_windows and ok_ci and ok_fdr
        any_winner |= winner
        print(f"  {arm:16s}: 3/3窗口={'Y' if ok_windows else 'N'} CI下界>0={'Y' if ok_ci else 'N'} "
              f"FDR={'Y' if ok_fdr else 'N'} -> {'通过' if winner else '不通过'}")
    verdict = "WINNER存在" if any_winner else "无 challenger 通过 -> 冻结 incumbent (risk_only)"
    print(f"\n最终判定: {verdict}")

    # DSR（对最优 challenger 的附加统计，仅在通过时相关）
    best = min(CHALLENGERS, key=lambda arm: results[arm]["p"])
    series = returns[best]
    t_count = series.size
    sr = sharpe(series) / math.sqrt(PERIODS_PER_YEAR)
    skew = float(((series - series.mean()) ** 3).mean() / series.std(ddof=1) ** 3)
    kurt = float(((series - series.mean()) ** 4).mean() / series.std(ddof=1) ** 4)
    expected_max = math.sqrt(2.0 * math.log(4))  # 4 arms 试验数
    sr_star = expected_max / math.sqrt(t_count)
    denom = math.sqrt(max(1e-18, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr))
    dsr = norm_cdf((sr - sr_star) * math.sqrt(t_count - 1) / denom)
    print(f"DSR({best}) = {dsr:.4f} (>0.95 才具 deflate 后显著性)")

    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    print(f"\nreport_sha256 = {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
