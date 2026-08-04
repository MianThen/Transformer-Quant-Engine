# trading_engine:C++ 低延迟交易引擎(Phase 2)
#
# 独立于回测引擎(cpp_engine)的可执行程序 + 静态库 + 测试。
# 可从顶层 CMake 用 add_subdirectory(trading_engine) 引入,也可单独构建:
#   cmake -S trading_engine -B build_te && cmake --build build_te

cmake_minimum_required(VERSION 3.15)
project(trading_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 核心静态库
add_library(te_core STATIC
    src/net/tcp_socket.cpp
    src/net/event_loop.cpp
    src/net/latency.cpp
    src/transport/tcp_transport.cpp
    src/adapter/mock_market_data_adapter.cpp
    src/adapter/mock_order_adapter.cpp
    src/feed/decoder.cpp
    src/feed/feed_handler.cpp
    src/oms/order_state.cpp
    src/oms/order_gateway.cpp
    src/runtime/runtime.cpp
)
target_include_directories(te_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(TARGET engine_common)
    target_link_libraries(te_core PUBLIC engine_common)
endif()

# Windows 需链接 Winsock
if(WIN32)
    target_link_libraries(te_core PUBLIC ws2_32 psapi)
endif()

# 演示可执行程序
add_executable(trading_engine src/main.cpp)
target_link_libraries(trading_engine PRIVATE te_core)

add_executable(replay_feed src/replay_feed.cpp)
target_link_libraries(replay_feed PRIVATE te_core)

# ---- 单元测试 ----
enable_testing()

add_executable(test_ring_buffer tests/test_ring_buffer.cpp)
target_link_libraries(test_ring_buffer PRIVATE te_core)
if(COMMAND qbt_static_test_runtime)
    qbt_static_test_runtime(test_ring_buffer)
endif()
add_test(NAME test_ring_buffer COMMAND test_ring_buffer)

add_executable(test_order_state tests/test_order_state.cpp)
target_link_libraries(test_order_state PRIVATE te_core)
if(COMMAND qbt_static_test_runtime)
    qbt_static_test_runtime(test_order_state)
endif()
add_test(NAME test_order_state COMMAND test_order_state)

add_executable(test_decoder tests/test_decoder.cpp)
target_link_libraries(test_decoder PRIVATE te_core)
if(COMMAND qbt_static_test_runtime)
    qbt_static_test_runtime(test_decoder)
endif()
add_test(NAME test_decoder COMMAND test_decoder)

add_executable(test_utilities tests/test_utilities.cpp)
target_link_libraries(test_utilities PRIVATE te_core)
if(COMMAND qbt_static_test_runtime)
    qbt_static_test_runtime(test_utilities)
endif()
add_test(NAME test_utilities COMMAND test_utilities)

add_executable(test_tcp_socket tests/test_tcp_socket.cpp)
target_link_libraries(test_tcp_socket PRIVATE te_core)
if(COMMAND qbt_static_test_runtime)
    qbt_static_test_runtime(test_tcp_socket)
endif()
add_test(NAME test_tcp_socket COMMAND test_tcp_socket)

add_executable(test_order_gateway tests/test_order_gateway.cpp)
target_link_libraries(test_order_gateway PRIVATE te_core)
if(COMMAND qbt_static_test_runtime)
    qbt_static_test_runtime(test_order_gateway)
endif()
add_test(NAME test_order_gateway COMMAND test_order_gateway)

add_executable(test_replay_feed tests/test_replay_feed.cpp)
target_link_libraries(test_replay_feed PRIVATE te_core)
if(COMMAND qbt_static_test_runtime)
    qbt_static_test_runtime(test_replay_feed)
endif()
add_test(NAME test_replay_feed COMMAND test_replay_feed)

add_executable(test_network_adapters tests/test_network_adapters.cpp)
target_link_libraries(test_network_adapters PRIVATE te_core)
if(COMMAND qbt_static_test_runtime)
    qbt_static_test_runtime(test_network_adapters)
endif()
add_test(NAME test_network_adapters COMMAND test_network_adapters)

add_executable(test_gateway_faults tests/test_gateway_faults.cpp)
target_link_libraries(test_gateway_faults PRIVATE te_core)
if(COMMAND qbt_static_test_runtime)
    qbt_static_test_runtime(test_gateway_faults)
endif()
add_test(NAME test_gateway_faults COMMAND test_gateway_faults)

if(COMMAND qbt_enable_warnings)
    foreach(target te_core trading_engine replay_feed test_ring_buffer test_order_state test_decoder test_utilities test_tcp_socket test_order_gateway test_replay_feed test_network_adapters test_gateway_faults)
        qbt_enable_warnings(${target})
    endforeach()
endif()

if(QBT_BUILD_BENCHMARKS)
    add_executable(te_benchmark benchmarks/benchmark_decoder.cpp)
    target_link_libraries(te_benchmark PRIVATE te_core)
    target_compile_definitions(te_benchmark PRIVATE
        QBT_BUILD_TYPE="$<CONFIG>"
        QBT_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
        QBT_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
        QBT_LTO_ENABLED=$<BOOL:${QBT_LTO_ACTIVE}>
    )
    if(COMMAND qbt_enable_warnings)
        qbt_enable_warnings(te_benchmark)
        qbt_enable_lto(te_benchmark)
        qbt_static_test_runtime(te_benchmark)
    endif()

    add_executable(te_benchmark_legacy benchmarks/benchmark_decoder.cpp)
    target_include_directories(te_benchmark_legacy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
    target_link_libraries(te_benchmark_legacy PRIVATE te_core)
    target_compile_definitions(te_benchmark_legacy PRIVATE
        TE_BENCHMARK_LEGACY=1
        QBT_BUILD_TYPE="$<CONFIG>"
        QBT_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
        QBT_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
        QBT_LTO_ENABLED=$<BOOL:${QBT_LTO_ACTIVE}>
    )
    if(COMMAND qbt_enable_warnings)
        qbt_enable_warnings(te_benchmark_legacy)
        qbt_enable_lto(te_benchmark_legacy)
        qbt_static_test_runtime(te_benchmark_legacy)
    endif()

    add_executable(te_pipeline_benchmark benchmarks/benchmark_pipeline.cpp)
    target_link_libraries(te_pipeline_benchmark PRIVATE te_core)
    target_compile_definitions(te_pipeline_benchmark PRIVATE
        QBT_BUILD_TYPE="$<CONFIG>"
        QBT_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
        QBT_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
        QBT_LTO_ENABLED=$<BOOL:${QBT_LTO_ACTIVE}>
    )
    if(COMMAND qbt_enable_warnings)
        qbt_enable_warnings(te_pipeline_benchmark)
        qbt_enable_lto(te_pipeline_benchmark)
        qbt_static_test_runtime(te_pipeline_benchmark)
    endif()
endif()

if(QBT_BUILD_FUZZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_executable(fuzz_decoder fuzz/fuzz_decoder.cpp)
    target_link_libraries(fuzz_decoder PRIVATE te_core)
    target_compile_options(fuzz_decoder PRIVATE -fsanitize=fuzzer)
    target_link_options(fuzz_decoder PRIVATE -fsanitize=fuzzer)

    add_executable(fuzz_wire_protocol fuzz/fuzz_wire_protocol.cpp)
    target_link_libraries(fuzz_wire_protocol PRIVATE te_core)
    target_compile_options(fuzz_wire_protocol PRIVATE -fsanitize=fuzzer)
    target_link_options(fuzz_wire_protocol PRIVATE -fsanitize=fuzzer)
endif()
