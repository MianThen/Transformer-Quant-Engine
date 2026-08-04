add_library(portfolio_math STATIC
    src/covariance.cpp
    src/risk_budget.cpp
    src/risk_model.cpp
    src/tail_risk.cpp
    src/multiple_testing.cpp
    src/quest.cpp
)
target_link_libraries(portfolio_math PUBLIC engine_common)
target_include_directories(portfolio_math PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(portfolio_math PUBLIC quant_math)
qbt_enable_warnings(portfolio_math)
qbt_enable_lto(portfolio_math)

add_executable(test_portfolio_math tests/test_portfolio_math.cpp)
target_link_libraries(test_portfolio_math PRIVATE portfolio_math)
qbt_enable_warnings(test_portfolio_math)
qbt_static_test_runtime(test_portfolio_math)
add_test(NAME test_portfolio_math COMMAND test_portfolio_math)

add_executable(test_multiple_testing tests/test_multiple_testing.cpp)
target_link_libraries(test_multiple_testing PRIVATE portfolio_math)
qbt_enable_warnings(test_multiple_testing)
qbt_static_test_runtime(test_multiple_testing)
add_test(NAME test_multiple_testing COMMAND test_multiple_testing)

add_executable(test_quest tests/test_quest.cpp)
target_link_libraries(test_quest PRIVATE portfolio_math)
qbt_enable_warnings(test_quest)
qbt_static_test_runtime(test_quest)
add_test(NAME test_quest COMMAND test_quest)
