add_library(portfolio_math STATIC
    src/covariance.cpp
    src/factor_model.cpp
    src/risk_budget.cpp
    src/risk_model.cpp
    src/rmt_denoising.cpp
    src/detoning.cpp
    src/hierarchical_linkage.cpp
    src/onc_partition.cpp
    src/cluster_model_artifact.cpp
    src/hrp_policy.cpp
    src/nco_policy.cpp
    src/reconciler.cpp
    src/policy_to_order.cpp
    src/portfolio_policy_artifact.cpp
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

add_executable(test_rmt_denoising tests/test_rmt_denoising.cpp)
target_link_libraries(test_rmt_denoising PRIVATE portfolio_math)
qbt_enable_warnings(test_rmt_denoising)
qbt_static_test_runtime(test_rmt_denoising)
add_test(NAME test_rmt_denoising COMMAND test_rmt_denoising)

add_executable(test_detoning tests/test_detoning.cpp)
target_link_libraries(test_detoning PRIVATE portfolio_math)
qbt_enable_warnings(test_detoning)
qbt_static_test_runtime(test_detoning)
add_test(NAME test_detoning COMMAND test_detoning)

add_executable(test_hierarchical_linkage tests/test_hierarchical_linkage.cpp)
target_link_libraries(test_hierarchical_linkage PRIVATE portfolio_math)
qbt_enable_warnings(test_hierarchical_linkage)
qbt_static_test_runtime(test_hierarchical_linkage)
add_test(NAME test_hierarchical_linkage COMMAND test_hierarchical_linkage)

add_executable(test_onc_partition tests/test_onc_partition.cpp)
target_link_libraries(test_onc_partition PRIVATE portfolio_math)
qbt_enable_warnings(test_onc_partition)
qbt_static_test_runtime(test_onc_partition)
add_test(NAME test_onc_partition COMMAND test_onc_partition)

add_executable(test_cluster_model_artifact tests/test_cluster_model_artifact.cpp)
target_link_libraries(test_cluster_model_artifact PRIVATE portfolio_math)
qbt_enable_warnings(test_cluster_model_artifact)
qbt_static_test_runtime(test_cluster_model_artifact)
add_test(NAME test_cluster_model_artifact COMMAND test_cluster_model_artifact)

add_executable(test_hrp_policy tests/test_hrp_policy.cpp)
target_link_libraries(test_hrp_policy PRIVATE portfolio_math)
qbt_enable_warnings(test_hrp_policy)
qbt_static_test_runtime(test_hrp_policy)
add_test(NAME test_hrp_policy COMMAND test_hrp_policy)

add_executable(test_nco_policy tests/test_nco_policy.cpp)
target_link_libraries(test_nco_policy PRIVATE portfolio_math)
qbt_enable_warnings(test_nco_policy)
qbt_static_test_runtime(test_nco_policy)
add_test(NAME test_nco_policy COMMAND test_nco_policy)

add_executable(test_reconciler tests/test_reconciler.cpp)
target_link_libraries(test_reconciler PRIVATE portfolio_math)
qbt_enable_warnings(test_reconciler)
qbt_static_test_runtime(test_reconciler)
add_test(NAME test_reconciler COMMAND test_reconciler)

add_executable(test_policy_to_order tests/test_policy_to_order.cpp)
target_link_libraries(test_policy_to_order PRIVATE portfolio_math)
qbt_enable_warnings(test_policy_to_order)
qbt_static_test_runtime(test_policy_to_order)
add_test(NAME test_policy_to_order COMMAND test_policy_to_order)

add_executable(test_portfolio_policy_artifact tests/test_portfolio_policy_artifact.cpp)
target_link_libraries(test_portfolio_policy_artifact PRIVATE portfolio_math)
qbt_enable_warnings(test_portfolio_policy_artifact)
qbt_static_test_runtime(test_portfolio_policy_artifact)
add_test(NAME test_portfolio_policy_artifact COMMAND test_portfolio_policy_artifact)

add_executable(test_factor_model tests/test_factor_model.cpp)
target_link_libraries(test_factor_model PRIVATE portfolio_math)
qbt_enable_warnings(test_factor_model)
qbt_static_test_runtime(test_factor_model)
add_test(NAME test_factor_model COMMAND test_factor_model)
