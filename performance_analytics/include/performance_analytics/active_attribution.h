#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace performance_analytics {

enum class AttributionStatus : std::uint8_t {
    OK,
    UNAVAILABLE,
    INVALID_INPUT,
    RECONCILIATION_FAILURE,
    DEGENERATE_LINKING,
};

enum class AttributionGroupKind : std::uint8_t {
    EQUITY_INDUSTRY,
    CASH,
    UNMAPPED,
};

enum class AttributionEffectKind : std::uint8_t {
    ALLOCATION,
    SELECTION,
    INTERACTION,
    OTHER,
};

struct AttributionEffectKey {
    AttributionGroupKind group_kind{AttributionGroupKind::EQUITY_INDUSTRY};
    std::uint64_t group_id{0};
    AttributionEffectKind effect_kind{AttributionEffectKind::ALLOCATION};

    [[nodiscard]] bool operator==(const AttributionEffectKey&) const = default;
    [[nodiscard]] bool operator<(const AttributionEffectKey& other) const noexcept;
};

struct AttributionEffectValue {
    AttributionEffectKey key;
    double value{0.0};
};

struct BrinsonGroupInput {
    AttributionGroupKind group_kind{AttributionGroupKind::EQUITY_INDUSTRY};
    std::uint64_t group_id{0};
    double portfolio_weight{0.0};
    double benchmark_weight{0.0};
    double portfolio_return{0.0};
    double benchmark_return{0.0};
};

struct BrinsonSpec {
    double reconciliation_tolerance{1e-10};
    std::uint64_t config_hash{0};
};

struct BrinsonProblem {
    double portfolio_return{0.0};
    double benchmark_return{0.0};
    bool benchmark_holdings_available{false};
    std::uint64_t benchmark_provenance_hash{0};
    std::uint64_t pit_classification_hash{0};
    std::span<const BrinsonGroupInput> groups;
    BrinsonSpec spec;
};

struct BrinsonGroupEffect {
    AttributionGroupKind group_kind{AttributionGroupKind::EQUITY_INDUSTRY};
    std::uint64_t group_id{0};
    double allocation{0.0};
    double selection{0.0};
    double interaction{0.0};
    double other{0.0};
    double total{0.0};
};

struct BrinsonResult {
    AttributionStatus status{AttributionStatus::INVALID_INPUT};
    double portfolio_return{0.0};
    double benchmark_return{0.0};
    double active_return{0.0};
    double effect_total{0.0};
    double reconciliation_residual{0.0};
    std::vector<BrinsonGroupEffect> groups;
    std::uint64_t artifact_hash{0};
};

[[nodiscard]] BrinsonResult compute_brinson_fachler(
    const BrinsonProblem& problem);
[[nodiscard]] std::vector<AttributionEffectValue> brinson_effect_values(
    const BrinsonResult& result);

struct MencheroPeriodInput {
    std::uint64_t period_id{0};
    double portfolio_return{0.0};
    double benchmark_return{0.0};
    std::span<const AttributionEffectValue> effects;
};

struct MencheroSpec {
    double equality_tolerance{1e-12};
    double degeneracy_tolerance{1e-18};
    double reconciliation_tolerance{1e-10};
    std::uint64_t config_hash{0};
};

struct MencheroPeriodCoefficient {
    std::uint64_t period_id{0};
    double active_return{0.0};
    double beta{0.0};
};

struct MencheroLinkedEffect {
    AttributionEffectKey key;
    double linked_value{0.0};
};

struct MencheroResult {
    AttributionStatus status{AttributionStatus::INVALID_INPUT};
    double cumulative_portfolio_return{0.0};
    double cumulative_benchmark_return{0.0};
    double cumulative_return_difference{0.0};
    double linking_constant{0.0};
    double linked_effect_total{0.0};
    double reconciliation_residual{0.0};
    bool used_equal_return_limit{false};
    std::vector<MencheroPeriodCoefficient> periods;
    std::vector<MencheroLinkedEffect> effects;
    std::uint64_t artifact_hash{0};
};

[[nodiscard]] MencheroResult menchero_link(
    std::span<const MencheroPeriodInput> periods,
    const MencheroSpec& spec);

}  // namespace performance_analytics
