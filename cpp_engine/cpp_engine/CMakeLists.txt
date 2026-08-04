# cpp_engine 子构建:pybind11 模块 + C++ 单元测试

# 核心引擎静态库(不含 bindings,便于测试直接链接)
add_library(qbt_core STATIC
    src/order.cpp
    src/order_book.cpp
    src/event_queue.cpp
    src/engine.cpp
    src/position.cpp
    src/pnl_tracker.cpp
)
target_include_directories(qbt_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(qbt_core PUBLIC engine_common)
qbt_enable_warnings(qbt_core)
qbt_enable_lto(qbt_core)

# pybind11 模块:import cpp_engine。开启时依赖缺失必须让配置失败，
# 避免出现“构建成功但没有 Python 模块”的假成功。
if(QBT_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(cpp_engine src/bindings.cpp src/arrow_bridge.cpp)
    target_link_libraries(cpp_engine PRIVATE qbt_core)
    if(QBT_ENABLE_PERFORMANCE_ANALYTICS)
        target_link_libraries(cpp_engine PRIVATE performance_analytics)
        target_compile_definitions(cpp_engine PRIVATE
            QBT_ENABLE_PERFORMANCE_ANALYTICS=1)
    endif()
    if(QBT_ENABLE_PORTFOLIO_MATH)
        target_link_libraries(cpp_engine PRIVATE portfolio_math)
        target_compile_definitions(cpp_engine PRIVATE QBT_ENABLE_PORTFOLIO_MATH=1)
    endif()
    if(MINGW)
        target_link_options(cpp_engine PRIVATE -static)
    endif()
    target_compile_definitions(cpp_engine PRIVATE
        QBT_BUILD_TYPE="$<CONFIG>"
        QBT_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
        QBT_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
        QBT_LTO_ENABLED=$<BOOL:${QBT_LTO_ACTIVE}>
    )
    qbt_enable_warnings(cpp_engine)
    qbt_enable_lto(cpp_engine)
    install(TARGETS cpp_engine LIBRARY DESTINATION .)
endif()

if(QBT_BUILD_FUZZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_executable(fuzz_arrow_schema fuzz/fuzz_arrow_schema.cpp)
    target_include_directories(fuzz_arrow_schema PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(fuzz_arrow_schema PRIVATE -fsanitize=fuzzer)
    target_link_options(fuzz_arrow_schema PRIVATE -fsanitize=fuzzer)
endif()

# ---- 单元测试 ----
enable_testing()

add_executable(test_order_book tests/test_order_book.cpp)
target_link_libraries(test_order_book PRIVATE qbt_core)
qbt_enable_warnings(test_order_book)
qbt_static_test_runtime(test_order_book)
add_test(NAME test_order_book COMMAND test_order_book)

add_executable(test_engine tests/test_engine.cpp)
target_link_libraries(test_engine PRIVATE qbt_core)
qbt_enable_warnings(test_engine)
qbt_static_test_runtime(test_engine)
add_test(NAME test_engine COMMAND test_engine)

if(QBT_ENABLE_PERFORMANCE_ANALYTICS)
    add_executable(test_period_contribution_engine_integration
        tests/test_period_contribution_engine_integration.cpp)
    target_link_libraries(test_period_contribution_engine_integration PRIVATE
        qbt_core performance_analytics)
    qbt_enable_warnings(test_period_contribution_engine_integration)
    qbt_static_test_runtime(test_period_contribution_engine_integration)
    add_test(NAME test_period_contribution_engine_integration
        COMMAND test_period_contribution_engine_integration)
    if(QBT_ENABLE_PORTFOLIO_MATH)
        add_executable(test_phase2a_policy_replay
            tests/test_phase2a_policy_replay.cpp)
        target_link_libraries(test_phase2a_policy_replay PRIVATE
            qbt_core performance_analytics portfolio_math)
        qbt_enable_warnings(test_phase2a_policy_replay)
        qbt_static_test_runtime(test_phase2a_policy_replay)
        add_test(NAME test_phase2a_policy_replay
            COMMAND test_phase2a_policy_replay)
    endif()
endif()

add_executable(test_pnl tests/test_pnl.cpp)
target_link_libraries(test_pnl PRIVATE qbt_core)
qbt_enable_warnings(test_pnl)
qbt_static_test_runtime(test_pnl)
add_test(NAME test_pnl COMMAND test_pnl)

add_executable(test_arrow_schema tests/test_arrow_schema.cpp)
target_include_directories(test_arrow_schema PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
qbt_enable_warnings(test_arrow_schema)
qbt_static_test_runtime(test_arrow_schema)
add_test(NAME test_arrow_schema COMMAND test_arrow_schema)

add_executable(qbt_dual_replay benchmarks/dual_replay.cpp)
target_link_libraries(qbt_dual_replay PRIVATE qbt_core)
qbt_enable_warnings(qbt_dual_replay)
qbt_static_test_runtime(qbt_dual_replay)
add_test(NAME qbt_dual_replay_self_test COMMAND qbt_dual_replay --self-test)

if(QBT_BUILD_BENCHMARKS)
    add_executable(qbt_benchmark benchmarks/benchmark_main.cpp)
    target_link_libraries(qbt_benchmark PRIVATE qbt_core)
    if(WIN32)
        target_link_libraries(qbt_benchmark PRIVATE psapi)
    endif()
    target_compile_definitions(qbt_benchmark PRIVATE
        QBT_BUILD_TYPE="$<CONFIG>"
        QBT_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
        QBT_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
        QBT_LTO_ENABLED=$<BOOL:${QBT_LTO_ACTIVE}>
    )
    qbt_enable_warnings(qbt_benchmark)
    qbt_enable_lto(qbt_benchmark)
    qbt_static_test_runtime(qbt_benchmark)

    add_executable(qbt_golden_replay benchmarks/golden_replay.cpp)
    target_link_libraries(qbt_golden_replay PRIVATE qbt_core)
    qbt_enable_warnings(qbt_golden_replay)
    qbt_enable_lto(qbt_golden_replay)
    qbt_static_test_runtime(qbt_golden_replay)
endif()
