#include "portfolio_math/risk_model.h"

#include <algorithm>
#include <cmath>

namespace portfolio_math {

bool valid_risk_preprocessor_spec(const RiskPreprocessorSpec& spec) noexcept {
    if (spec.lookback_observations < 2 ||
        !std::isfinite(spec.concentration_ratio_guard) ||
        spec.concentration_ratio_guard < 0.0 || spec.concentration_ratio_guard >= 0.5 ||
        !std::isfinite(spec.eigenvalue_floor) || spec.eigenvalue_floor < 0.0 ||
        spec.balanced_panel_policy_hash == 0 || spec.config_hash == 0) {
        return false;
    }
    if (spec.official_estimator == CovarianceEstimator::LEDOIT_WOLF_NONLINEAR_QUEST) {
        return spec.covariance_loss == CovarianceLossProfile::MINIMUM_VARIANCE &&
            spec.uniform_observation_weights && spec.concentration_ratio_guard > 0.0 &&
            spec.quest_solver_spec_hash != 0;
    }
    if (spec.quest_solver_spec_hash != 0) return false;
    if (spec.official_estimator ==
        CovarianceEstimator::LEDOIT_WOLF_LINEAR_CONSTANT_CORRELATION) {
        return spec.covariance_loss == CovarianceLossProfile::FROBENIUS;
    }
    if (spec.official_estimator == CovarianceEstimator::SAMPLE) {
        return spec.covariance_loss == CovarianceLossProfile::NOT_APPLICABLE;
    }
    return true;
}

bool valid_dense_risk_model(
    const DenseRiskModelView& model,
    engine_common::TimestampNs decision_at,
    double symmetry_tolerance) {
    const bool regular_branch = model.effective_observations > model.symbols.size() &&
        model.dimensional_branch == DimensionalBranch::REGULAR_P_LT_N;
    const bool singular_branch = model.effective_observations < model.symbols.size() &&
        model.dimensional_branch == DimensionalBranch::SINGULAR_P_GT_N;
    const bool is_quest = model.estimator == CovarianceEstimator::LEDOIT_WOLF_NONLINEAR_QUEST;
    const bool valid_branch = is_quest
        ? (regular_branch || singular_branch)
        : model.dimensional_branch == DimensionalBranch::NOT_APPLICABLE;
    const bool valid_identity =
        (model.estimator == CovarianceEstimator::SAMPLE &&
         model.loss_profile == CovarianceLossProfile::NOT_APPLICABLE) ||
        (model.estimator == CovarianceEstimator::LEDOIT_WOLF_LINEAR_CONSTANT_CORRELATION &&
         model.loss_profile == CovarianceLossProfile::FROBENIUS) ||
        (is_quest && model.loss_profile == CovarianceLossProfile::MINIMUM_VARIANCE) ||
        (model.estimator != CovarianceEstimator::SAMPLE &&
         model.estimator != CovarianceEstimator::LEDOIT_WOLF_LINEAR_CONSTANT_CORRELATION &&
         !is_quest);
    const double expected_ratio = static_cast<double>(model.symbols.size()) /
                                  static_cast<double>(model.effective_observations);
    return decision_at > 0 && model.fit_start > 0 && model.fit_start <= model.fit_end &&
        model.fit_end <= model.available_at && model.available_at <= decision_at &&
        model.artifact_hash != 0 && model.symbol_mapping_hash != 0 &&
        model.balanced_panel_policy_hash != 0 && model.effective_observations > 0 &&
        model.concentration_ratio > 0.0 && std::isfinite(model.concentration_ratio) &&
        std::abs(model.concentration_ratio - expected_ratio) <= 1e-12 &&
        valid_branch && valid_identity &&
        !model.symbols.empty() &&
        std::is_sorted(model.symbols.begin(), model.symbols.end()) &&
        std::adjacent_find(model.symbols.begin(), model.symbols.end()) == model.symbols.end() &&
        model.covariance.rows == model.symbols.size() &&
        model.covariance.cols == model.symbols.size() &&
        quant_math::validate_symmetric(model.covariance, symmetry_tolerance).ok;
}

}  // namespace portfolio_math
