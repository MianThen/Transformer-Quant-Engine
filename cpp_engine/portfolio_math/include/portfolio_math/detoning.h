#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "quant_math/matrix.h"

namespace portfolio_math {

struct DetoningSpec {
  std::uint32_t components{1};
  double eigenvalue_floor{1e-12};
  double psd_tolerance{1e-10};
  double symmetry_tolerance{1e-10};
};

enum class DetoningStatus {
  OK,
  INVALID_INPUT,
  NUMERICAL_FAILURE,
  DEGENERATE_OUTPUT,
};

struct DetoningDiagnostics {
  std::uint32_t removed_components{0};
  std::vector<double> raw_eigenvalues;
  std::vector<double> detoned_eigenvalues;
  double input_trace{0.0};
  double output_trace{0.0};
  double trace_drift{0.0};
  double minimum_output_eigenvalue{0.0};
  double psd_repair_amount{0.0};
  bool eligible_for_official_risk{false};
};

struct DetoningResult {
  DetoningStatus status{DetoningStatus::INVALID_INPUT};
  quant_math::DenseMatrix correlation;
  DetoningDiagnostics diagnostics;
};

[[nodiscard]] bool valid_detoning_spec(const DetoningSpec& spec) noexcept;

[[nodiscard]] DetoningResult detone_correlation(
    quant_math::MatrixView correlation, DetoningSpec spec = {});

}  // namespace portfolio_math
