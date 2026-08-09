#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "portfolio_math/covariance.h"
#include "quant_math/matrix.h"

namespace portfolio_math {

struct RmtDenoisingSpec {
  double edge_tolerance{1e-10};
  double eigenvalue_floor{1e-12};
  double psd_tolerance{1e-10};
  double targeted_shrinkage_intensity{0.5};
};

struct RmtDenoisingResult {
  CovarianceResult covariance;
  RmtDenoisingDiagnostics diagnostics;
};

[[nodiscard]] bool valid_rmt_denoising_spec(
    const RmtDenoisingSpec& spec) noexcept;

[[nodiscard]] RmtDenoisingResult rmt_constant_residual_denoising(
    quant_math::MatrixView returns,
    RmtDenoisingSpec spec = {});

[[nodiscard]] RmtDenoisingResult rmt_targeted_shrinkage_denoising(
    quant_math::MatrixView returns,
    RmtDenoisingSpec spec = {});

}  // namespace portfolio_math
