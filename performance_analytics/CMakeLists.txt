add_library(performance_analytics STATIC
    src/active_attribution.cpp
    src/factor_attribution_v2.cpp
    src/implementation_shortfall.cpp
    src/infots_replay.cpp
    src/period_contribution_coordinator.cpp
    src/period_contribution_replay_sink.cpp
    src/performance_spec.cpp
    src/return_analysis.cpp
    src/return_ledger.cpp
    src/security_contribution.cpp
    src/shortfall_replay_adapter.cpp
    src/shortfall_replay_sink.cpp
)
target_include_directories(performance_analytics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(performance_analytics PUBLIC engine_common)
if(QBT_ENABLE_PORTFOLIO_MATH)
    target_sources(performance_analytics PRIVATE
        src/policy_family_governance.cpp)
    target_link_libraries(performance_analytics PUBLIC portfolio_math)
endif()
qbt_enable_warnings(performance_analytics)
qbt_enable_lto(performance_analytics)

add_executable(test_performance_analytics tests/test_return_ledger.cpp)
target_link_libraries(test_performance_analytics PRIVATE performance_analytics)
qbt_enable_warnings(test_performance_analytics)
qbt_static_test_runtime(test_performance_analytics)
add_test(NAME test_performance_analytics COMMAND test_performance_analytics)

add_executable(test_return_analysis tests/test_return_analysis.cpp)
target_link_libraries(test_return_analysis PRIVATE performance_analytics)
qbt_enable_warnings(test_return_analysis)
qbt_static_test_runtime(test_return_analysis)
add_test(NAME test_return_analysis COMMAND test_return_analysis)

add_executable(test_infots_replay tests/test_infots_replay.cpp)
target_link_libraries(test_infots_replay PRIVATE performance_analytics)
qbt_enable_warnings(test_infots_replay)
qbt_static_test_runtime(test_infots_replay)
add_test(NAME test_infots_replay COMMAND test_infots_replay)

add_executable(test_implementation_shortfall tests/test_implementation_shortfall.cpp)
target_link_libraries(test_implementation_shortfall PRIVATE performance_analytics)
qbt_enable_warnings(test_implementation_shortfall)
qbt_static_test_runtime(test_implementation_shortfall)
add_test(NAME test_implementation_shortfall COMMAND test_implementation_shortfall)

add_executable(test_shortfall_replay_adapter tests/test_shortfall_replay_adapter.cpp)
target_link_libraries(test_shortfall_replay_adapter PRIVATE performance_analytics)
qbt_enable_warnings(test_shortfall_replay_adapter)
qbt_static_test_runtime(test_shortfall_replay_adapter)
add_test(NAME test_shortfall_replay_adapter COMMAND test_shortfall_replay_adapter)

add_executable(test_security_contribution tests/test_security_contribution.cpp)
target_link_libraries(test_security_contribution PRIVATE performance_analytics)
qbt_enable_warnings(test_security_contribution)
qbt_static_test_runtime(test_security_contribution)
add_test(NAME test_security_contribution COMMAND test_security_contribution)

add_executable(test_period_contribution_coordinator
    tests/test_period_contribution_coordinator.cpp)
target_link_libraries(test_period_contribution_coordinator PRIVATE
    performance_analytics)
qbt_enable_warnings(test_period_contribution_coordinator)
qbt_static_test_runtime(test_period_contribution_coordinator)
add_test(NAME test_period_contribution_coordinator
    COMMAND test_period_contribution_coordinator)

add_executable(test_active_attribution tests/test_active_attribution.cpp)
target_link_libraries(test_active_attribution PRIVATE performance_analytics)
qbt_enable_warnings(test_active_attribution)
qbt_static_test_runtime(test_active_attribution)
add_test(NAME test_active_attribution COMMAND test_active_attribution)

add_executable(test_factor_attribution_v2 tests/test_factor_attribution_v2.cpp)
target_link_libraries(test_factor_attribution_v2 PRIVATE performance_analytics)
qbt_enable_warnings(test_factor_attribution_v2)
qbt_static_test_runtime(test_factor_attribution_v2)
add_test(NAME test_factor_attribution_v2 COMMAND test_factor_attribution_v2)

if(QBT_ENABLE_PORTFOLIO_MATH)
    add_executable(test_policy_family_governance
        tests/test_policy_family_governance.cpp)
    target_link_libraries(test_policy_family_governance PRIVATE
        performance_analytics)
    qbt_enable_warnings(test_policy_family_governance)
    qbt_static_test_runtime(test_policy_family_governance)
    add_test(NAME test_policy_family_governance
        COMMAND test_policy_family_governance)

    add_executable(phase4b_governance_reference
        tests/phase4b_governance_reference.cpp)
    target_link_libraries(phase4b_governance_reference PRIVATE
        performance_analytics)
    qbt_enable_warnings(phase4b_governance_reference)
    qbt_static_test_runtime(phase4b_governance_reference)
    add_test(NAME test_phase4b_governance_reference
        COMMAND phase4b_governance_reference --output
            "${CMAKE_CURRENT_BINARY_DIR}/phase4b-governance-reference/phase4b_governance_reference.json")

    add_executable(phase4a_posterior_direct_reference
        tests/phase4a_posterior_direct_reference.cpp)
    target_link_libraries(phase4a_posterior_direct_reference PRIVATE
        performance_analytics)
    qbt_enable_warnings(phase4a_posterior_direct_reference)
    qbt_static_test_runtime(phase4a_posterior_direct_reference)
    add_test(NAME test_phase4a_posterior_direct_reference
        COMMAND phase4a_posterior_direct_reference --output
            "${CMAKE_CURRENT_BINARY_DIR}/phase4a-posterior-direct-reference/phase4a_posterior_direct_reference.json")
endif()

# ---------------------------------------------------------------------------
# MC-GATE-VALIDATION v1.4 施工产物（本地量化模型按分卡方案生成）
#   · 10 个 header-only 自检：每卡一个可执行文件，各自一个 selftest 宏
#   · 1 个 smoke 驱动：DGP -> baselines -> decide -> 记录 -> gzip 落盘
# 全部零第三方依赖。头文件之间用同目录裸名互相包含，故统一加
# include/performance_analytics 到包含路径。
# mc_gate_rng.h 用 unsigned __int128 实现 PCG64，会触发 -Wpedantic，
# 因此仅对这几个目标关掉 -Wpedantic（其余告警保持开启）。
# ---------------------------------------------------------------------------
set(QBT_MC_GATE_HEADER_SELFTESTS
    rng dgp truth baselines decide adapter runner scorer output onc manifest sha256 replay cost wall threshold seed)

foreach(mc_gate_name IN LISTS QBT_MC_GATE_HEADER_SELFTESTS)
    add_executable(test_mc_gate_${mc_gate_name}
        tests/mc_gate/test_mc_gate_${mc_gate_name}.cpp)
    target_include_directories(test_mc_gate_${mc_gate_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/performance_analytics)
    qbt_enable_warnings(test_mc_gate_${mc_gate_name})
    # -Wno-pedantic: mc_gate_rng.h 用 unsigned __int128 实现 PCG64
    # -Wno-unused-variable: C01 的 selftest 内有一个未使用局部变量（模型产物）；
    #   重新生成该头需要连带重跑它全部 golden 向量，暂以目标级抑制，仅影响 selftest 脚手架，
    #   库本体（未定义 MC_GATE_*_SELFTEST 时）不会产生该告警。
    target_compile_options(test_mc_gate_${mc_gate_name} PRIVATE
        -Wno-pedantic -Wno-unused-variable)
    qbt_static_test_runtime(test_mc_gate_${mc_gate_name})
    add_test(NAME test_mc_gate_${mc_gate_name}
        COMMAND test_mc_gate_${mc_gate_name})
endforeach()

add_executable(mc_gate_validation apps/mc_gate_validation.cpp)
target_include_directories(mc_gate_validation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/performance_analytics)
qbt_enable_warnings(mc_gate_validation)
target_compile_options(mc_gate_validation PRIVATE -Wno-pedantic)
qbt_static_test_runtime(mc_gate_validation)

# §5.8 COSTP：在目标机实测 ĉ(T,N) 与分项占比（不注册为 ctest，需人工在目标机跑）
add_executable(mc_gate_costp apps/mc_gate_costp.cpp)
target_include_directories(mc_gate_costp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/performance_analytics)
qbt_enable_warnings(mc_gate_costp)
target_compile_options(mc_gate_costp PRIVATE -Wno-pedantic)
qbt_static_test_runtime(mc_gate_costp)

# F1/F2 诊断探针：GATE 的实测 FPR 与功率曲线（用于 α 网格与 canary 定档，不注册 ctest）
foreach(mc_gate_probe fpr power)
    add_executable(mc_gate_${mc_gate_probe} apps/mc_gate_${mc_gate_probe}.cpp)
    target_include_directories(mc_gate_${mc_gate_probe} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/performance_analytics)
    qbt_enable_warnings(mc_gate_${mc_gate_probe})
    target_compile_options(mc_gate_${mc_gate_probe} PRIVATE -Wno-pedantic)
    qbt_static_test_runtime(mc_gate_${mc_gate_probe})
endforeach()

# 集成驱动 v2（C17）：plan -> cost_profile -> metrics(含 cost_profile) -> records -> replay
# 只构建、不注册 ctest：它会写 /tmp/qgate/C17 下的 smoke 产物，需人工运行。
add_executable(mc_gate_run apps/mc_gate_run.cpp)
target_include_directories(mc_gate_run PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/performance_analytics)
qbt_enable_warnings(mc_gate_run)
target_compile_options(mc_gate_run PRIVATE -Wno-pedantic)
qbt_static_test_runtime(mc_gate_run)
