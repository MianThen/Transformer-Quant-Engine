cmake_minimum_required(VERSION 3.15)
project(quant_backtester_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(QBT_BUILD_PYTHON "Build the cpp_engine Python extension" ON)
option(QBT_BUILD_LIVE_ENGINE "Build the experimental live trading engine" OFF)
option(QBT_BUILD_BENCHMARKS "Build deterministic C++ benchmarks" OFF)
option(QBT_BUILD_FUZZERS "Build libFuzzer targets" OFF)
option(QBT_ENABLE_LTO "Enable interprocedural optimization for Release builds" OFF)
option(QBT_ENABLE_WARNINGS "Enable strict compiler warnings" ON)
option(QBT_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(QBT_ENABLE_ML "Build the optional model and strategy runtimes" OFF)
option(QBT_BUILD_ML_TESTS "Build optional ML runtime tests" OFF)
option(QBT_ENABLE_PORTFOLIO_MATH "Build optional quantitative and portfolio math" OFF)
option(QBT_ENABLE_CONVEX_SOLVER "Enable optional convex solver adapters" OFF)
option(QBT_ENABLE_PERFORMANCE_ANALYTICS "Build optional performance analytics" OFF)
set(QBT_ML_BACKEND "onnxruntime" CACHE STRING "ML inference backend (mock or onnxruntime)")
set_property(CACHE QBT_ML_BACKEND PROPERTY STRINGS mock onnxruntime)
option(QBT_ML_ENABLE_CUDA "Enable the ONNX Runtime CUDA provider" OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

enable_testing()

function(qbt_enable_warnings target)
    if(NOT QBT_ENABLE_WARNINGS)
        return()
    endif()
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
        if(QBT_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        if(QBT_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
endfunction()

set(QBT_LTO_ACTIVE OFF)
if(QBT_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT qbt_ipo_supported OUTPUT qbt_ipo_error)
    if(qbt_ipo_supported)
        set(QBT_LTO_ACTIVE ON)
    else()
        message(WARNING "LTO requested but unavailable; report will record lto=false: ${qbt_ipo_error}")
    endif()
endif()

function(qbt_enable_lto target)
    if(NOT QBT_LTO_ACTIVE)
        return()
    endif()
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endfunction()

function(qbt_static_test_runtime target)
    if(MINGW)
        target_link_options(${target} PRIVATE -static)
    endif()
endfunction()

if(QBT_BUILD_BENCHMARKS AND NOT CMAKE_CONFIGURATION_TYPES AND
   NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(FATAL_ERROR "Benchmarks must be configured with CMAKE_BUILD_TYPE=Release")
endif()

add_subdirectory(engine_common)

if(QBT_ENABLE_PORTFOLIO_MATH)
    include(FetchContent)
    FetchContent_Declare(
        qbt_eigen_source
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()
    FetchContent_GetProperties(qbt_eigen_source)
    if(NOT qbt_eigen_source_POPULATED)
        FetchContent_Populate(qbt_eigen_source)
    endif()
    add_library(qbt_eigen INTERFACE)
    target_include_directories(qbt_eigen SYSTEM INTERFACE ${qbt_eigen_source_SOURCE_DIR})

    add_subdirectory(quant_math)
    add_subdirectory(portfolio_math)
endif()

if(QBT_ENABLE_PERFORMANCE_ANALYTICS)
    add_subdirectory(performance_analytics)
endif()

if(QBT_ENABLE_ML)
    add_subdirectory(ml_runtime)
endif()

# ── C++ 核心 1:回测引擎(pybind11 模块 cpp_engine + 单元测试)──
# 需要 pybind11:`pip install pybind11`,或用 -Dpybind11_DIR=... 指定。
add_subdirectory(cpp_engine)

if(QBT_ENABLE_ML)
    add_subdirectory(strategy_runtime)
    if(QBT_BUILD_PYTHON)
        target_link_libraries(cpp_engine PRIVATE strategy_runtime)
        target_compile_definitions(cpp_engine PRIVATE QBT_ENABLE_ML=1)
        if(QBT_ML_BACKEND STREQUAL "onnxruntime")
            target_compile_definitions(cpp_engine PRIVATE QBT_ML_ONNXRUNTIME=1)
        endif()
    endif()
endif()

# ── C++ 核心 2:低延迟交易引擎(Phase 2,独立可执行 + 测试)──
if(QBT_BUILD_LIVE_ENGINE)
    add_subdirectory(trading_engine)
endif()
