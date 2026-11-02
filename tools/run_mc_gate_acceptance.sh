#!/usr/bin/env bash
# MC-GATE-VALIDATION v1.4 验收执行器
#
# 四步：
#   A. L0 自检 —— 用归档的"已知正确样本"验证验收脚本本身能判 PASS
#   B. 构建     —— 10 个 selftest 目标 + 4 个 app
#   C. 单测     —— ctest 跑 10 项 selftest
#   D. 驱动     —— 跑 smoke 驱动与 3 个诊断探针，检查各自的完成标记
#
# 用法：
#   tools/run_mc_gate_acceptance.sh [--skip-build] [--build-dir DIR]
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build/mc-gate}"
SKIP_BUILD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-build) SKIP_BUILD=1; shift ;;
    --build-dir)  BUILD_DIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

CARDS="$REPO_ROOT/configs/mc/cards"
PASS=0; FAIL=0
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); }

echo "== A. L0 自检：验收脚本 vs 归档期望样本 =="
l0() { # <name> <awk> <sample> <marker>
  if awk -f "$2" "$3" 2>/dev/null | grep -qE "^$4"; then ok "$1"; else bad "$1"; fi
}
l0 "C06 记录/分片不变量" "$CARDS/verify/c06_verify.awk" "$CARDS/expected/c06_sample_ok.txt" "C06_PASS"
l0 "C07 Wilson/McNemar/四态" "$CARDS/verify/c07_verify.awk" "$CARDS/expected/c07_sample_ok.txt" "C07_PASS"
l0 "C08 gzip 头/JSONL/转义" "$CARDS/verify/c08_verify.awk" "$CARDS/expected/c08_expected.txt" "C08_AWK_PASS"
l0 "C09 ONC 四态与择优" "$CARDS/verify/c09_verify.awk" "$CARDS/expected/c09_expected.txt" "C09_PASS"
l0 "C10 网格推导/分桶降级" "$CARDS/verify/c10_verify.awk" "$CARDS/expected/c10_expected.txt" "C10_PASS"
l0 "C11 分片归并/聚合" "$CARDS/verify/c11_verify.awk" "$CARDS/expected/c11_expected.txt" "C11_PASS"
l0 "C13 manifest/降档/metrics" "$CARDS/verify/c13_verify.awk" "$CARDS/expected/c13_expected.txt" "C13_PASS"
l0 "C14 SHA-256 golden 向量" "$CARDS/verify/c14_verify.awk" "$CARDS/expected/c14_expected.txt" "C14_PASS"
l0 "C15 replay 每数据集一文件" "$CARDS/verify/c15_verify.awk" "$CARDS/expected/c15_expected.txt" "C15_PASS"
l0 "C16 cost_profile 汇总" "$CARDS/verify/c16_verify.awk" "$CARDS/expected/c16_expected.txt" "C16_PASS"
l0 "C18 wall_by_rule 序列化" "$CARDS/verify/c18_verify.awk" "$CARDS/expected/c18_expected.txt" "C18_PASS"
l0 "C19 双轨门槛" "$CARDS/verify/c19_verify.awk" "$CARDS/expected/c19_expected.txt" "C19_PASS"
l0 "C20 图的可追溯性" "$CARDS/verify/c20_verify.awk" "$CARDS/expected/c20_expected.txt" "C20_PASS"
l0 "C21 Python 独立复算" "$CARDS/verify/c21_verify.awk" "$CARDS/expected/c21_expected.txt" "C21_PASS"
l0 "C22 §3.6 种子树向量" "$CARDS/verify/c22_verify.awk" "$CARDS/expected/c22_expected.txt" "C22_PASS"
l0 "C23 面板接种子树" "$CARDS/verify/c23_verify.awk" "$CARDS/expected/c23_expected.txt" "C23_PASS"

if [ "$SKIP_BUILD" -eq 0 ]; then
  echo
  echo "== B. 构建（${BUILD_DIR}）=="
  if cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DQBT_ENABLE_PERFORMANCE_ANALYTICS=ON \
        -DQBT_BUILD_PYTHON=OFF >/dev/null 2>&1 \
     && cmake --build "$BUILD_DIR" -j "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)" >/dev/null 2>&1; then
    ok "cmake configure + build"
  else
    bad "cmake configure + build"; echo "  放弃后续步骤"; exit 1
  fi
fi

echo
echo "== C. 单测：ctest -R mc_gate =="
CT="$(cd "$BUILD_DIR" && ctest -R mc_gate 2>&1 | tail -4)"
if echo "$CT" | grep -q "100% tests passed"; then
  ok "18 个 selftest 全过"
else
  bad "selftest"; echo "$CT" | sed 's/^/    /'
fi

echo
echo "== D. 驱动与探针 =="
APP="$BUILD_DIR/performance_analytics"
mkdir -p /tmp/qgate/C12
run_marker() { # <name> <binary> <marker>
  if [ -x "$2" ] && "$2" 2>/dev/null | grep -qE "^$3"; then ok "$1"; else bad "$1"; fi
}
run_marker "smoke 驱动（1200 条 + 落盘）" "$APP/mc_gate_validation" "SMOKE_DONE"
run_marker "COSTP 成本测量（14 点）"       "$APP/mc_gate_costp"      "COSTP_DONE"
run_marker "FPR 探针（3×3 网格）"          "$APP/mc_gate_fpr"        "FPR_DONE"
run_marker "功率探针（4×6 曲线）"          "$APP/mc_gate_power"      "POWER_DONE"
mkdir -p /tmp/qgate/C17/replay
run_marker "集成驱动 v2（plan/cost/metrics/replay）" "$APP/mc_gate_run"      "RUN_DONE"
if python3 "$REPO_ROOT/tools/mc_gate_plot.py" 2>/dev/null | grep -qx "PLOT_DONE"; then ok "C20 图追溯（tools/mc_gate_plot.py）"; else bad "C20 图追溯"; fi
if python3 "$REPO_ROOT/tools/mc_gate_grid_check.py" 2>/dev/null | grep -qx "GRID_MATCH 1"; then ok "AC-01 网格推导（269/202/617000）"; else bad "AC-01 网格推导（269/202/617000）"; fi

echo
echo "---------------------------------------------"
printf '合计：PASS %d / %d\n' "$PASS" "$((PASS+FAIL))"
[ "$FAIL" -eq 0 ] || exit 1
