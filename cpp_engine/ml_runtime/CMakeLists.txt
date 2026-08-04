set(QBT_ML_RUNTIME_SOURCES
    src/bar_v1_feature_pipeline.cpp
    src/feature_window_store.cpp
    src/mock_inference_backend.cpp
)
set(QBT_ML_RUNTIME_LIBRARIES engine_common)

if(QBT_ML_BACKEND STREQUAL "onnxruntime")
    if(QBT_ML_ENABLE_CUDA)
        message(FATAL_ERROR "The first ONNX Runtime integration supports CPU only")
    endif()
    find_package(onnxruntime CONFIG QUIET)
    if(TARGET onnxruntime::onnxruntime)
        set(QBT_ONNXRUNTIME_TARGET onnxruntime::onnxruntime)
    elseif(TARGET onnxruntime)
        set(QBT_ONNXRUNTIME_TARGET onnxruntime)
    else()
        find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
            HINTS ${ONNXRUNTIME_ROOT} ENV ONNXRUNTIME_ROOT
            PATH_SUFFIXES include include/onnxruntime/core/session)
        find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
            HINTS ${ONNXRUNTIME_ROOT} ENV ONNXRUNTIME_ROOT
            PATH_SUFFIXES lib lib64)
        if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
            message(FATAL_ERROR
                "QBT_ML_BACKEND=onnxruntime requires an onnxruntime CMake package "
                "or ONNXRUNTIME_ROOT containing include/ and lib/")
        endif()
        add_library(qbt_onnxruntime UNKNOWN IMPORTED)
        set_target_properties(qbt_onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")
        set(QBT_ONNXRUNTIME_TARGET qbt_onnxruntime)
    endif()
    list(APPEND QBT_ML_RUNTIME_SOURCES src/onnx_runtime_backend.cpp)
    list(APPEND QBT_ML_RUNTIME_LIBRARIES ${QBT_ONNXRUNTIME_TARGET})
elseif(NOT QBT_ML_BACKEND STREQUAL "mock")
    message(FATAL_ERROR "QBT_ML_BACKEND must be mock or onnxruntime")
endif()

add_library(ml_runtime STATIC ${QBT_ML_RUNTIME_SOURCES})
target_include_directories(ml_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ml_runtime PUBLIC ${QBT_ML_RUNTIME_LIBRARIES})
qbt_enable_warnings(ml_runtime)
qbt_enable_lto(ml_runtime)

if(QBT_BUILD_ML_TESTS)
    add_executable(test_ml_runtime tests/test_ml_runtime.cpp)
    target_link_libraries(test_ml_runtime PRIVATE ml_runtime)
    qbt_enable_warnings(test_ml_runtime)
    qbt_static_test_runtime(test_ml_runtime)
    add_test(NAME test_ml_runtime COMMAND test_ml_runtime)
endif()
