add_library(strategy_runtime STATIC
    src/model_strategy_runtime.cpp
    src/portfolio_policy.cpp
    src/risk_manager.cpp
)
target_include_directories(strategy_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strategy_runtime PUBLIC engine_common ml_runtime)
if(QBT_ENABLE_PORTFOLIO_MATH)
    target_sources(strategy_runtime PRIVATE src/research_portfolio_policy.cpp)
    target_link_libraries(strategy_runtime PUBLIC portfolio_math)
    target_compile_definitions(strategy_runtime PUBLIC QBT_ENABLE_PORTFOLIO_MATH=1)
endif()
qbt_enable_warnings(strategy_runtime)
qbt_enable_lto(strategy_runtime)

if(QBT_BUILD_ML_TESTS)
    add_executable(test_model_strategy_runtime tests/test_model_strategy_runtime.cpp)
    target_link_libraries(test_model_strategy_runtime PRIVATE strategy_runtime qbt_core)
    qbt_enable_warnings(test_model_strategy_runtime)
    qbt_static_test_runtime(test_model_strategy_runtime)
    add_test(NAME test_model_strategy_runtime COMMAND test_model_strategy_runtime)
    if(QBT_ENABLE_PORTFOLIO_MATH)
        add_executable(test_research_portfolio_policy tests/test_research_portfolio_policy.cpp)
        target_link_libraries(test_research_portfolio_policy PRIVATE strategy_runtime)
        qbt_enable_warnings(test_research_portfolio_policy)
        qbt_static_test_runtime(test_research_portfolio_policy)
add_test(NAME test_research_portfolio_policy COMMAND test_research_portfolio_policy)

if(QBT_ENABLE_PERFORMANCE_ANALYTICS)
    add_executable(test_shortfall_engine_integration
        tests/test_shortfall_engine_integration.cpp)
    target_link_libraries(test_shortfall_engine_integration PRIVATE
        strategy_runtime qbt_core performance_analytics)
    qbt_enable_warnings(test_shortfall_engine_integration)
    qbt_static_test_runtime(test_shortfall_engine_integration)
    add_test(NAME test_shortfall_engine_integration
        COMMAND test_shortfall_engine_integration)
endif()
    endif()
endif()
