add_library(quant_math STATIC
    src/matrix.cpp
)
target_include_directories(quant_math PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(quant_math PUBLIC qbt_eigen)
qbt_enable_warnings(quant_math)
qbt_enable_lto(quant_math)

add_executable(test_quant_math tests/test_matrix.cpp)
target_link_libraries(test_quant_math PRIVATE quant_math)
qbt_enable_warnings(test_quant_math)
qbt_static_test_runtime(test_quant_math)
add_test(NAME test_quant_math COMMAND test_quant_math)
