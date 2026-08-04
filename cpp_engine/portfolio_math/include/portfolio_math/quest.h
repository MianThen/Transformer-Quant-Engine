#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace portfolio_math {

enum class QuestStatus : std::uint8_t {
  OK,
  INVALID_INPUT,
  CONCENTRATION_RATIO_TOO_CLOSE_TO_ONE,
  NUMERICAL_FAILURE,
  MAX_ITERATIONS,
};

struct QuestSpec {
  std::uint32_t max_root_iterations{160};
  std::uint32_t max_inverse_iterations{80};
  std::uint32_t max_line_search_steps{16};
  double root_tolerance{1e-12};
  // Dimension-normalized squared residual relative to the mean spectrum.
  // Observed spectra need projection onto the QuEST image, so zero is not
  // generally attainable.
  double objective_tolerance{1e-6};
  double step_tolerance{1e-8};
  double finite_difference_step{1e-5};
  double initial_damping{1e-4};
  double concentration_ratio_guard{1e-6};
  double eigenvalue_floor{1e-12};
  double angle_row_mass_tolerance{1e-6};
};

struct QuestSupportInterval {
  double u_left{0.0};
  double u_right{0.0};
  double x_left{0.0};
  double x_right{0.0};
  std::size_t population_eigenvalue_count{0};
};

struct QuestDiagnostics {
  std::vector<QuestSupportInterval> support;
  std::size_t grid_size{0};
  std::uint32_t iterations{0};
  double concentration_ratio{0.0};
  double atom_at_zero{0.0};
  double maximum_raw_mass_error{0.0};
  double objective{0.0};
  double maximum_forward_residual{0.0};
};

struct QuestForwardResult {
  QuestStatus status{QuestStatus::INVALID_INPUT};
  std::vector<double> sample_eigenvalues;
  QuestDiagnostics diagnostics;
};

struct QuestInverseResult {
  QuestStatus status{QuestStatus::INVALID_INPUT};
  std::vector<double> population_eigenvalues;
  QuestDiagnostics diagnostics;
};

struct QuestStieltjesResult {
  QuestStatus status{QuestStatus::INVALID_INPUT};
  std::complex<double> value{0.0, 0.0};
  double mapped_sample_eigenvalue{0.0};
  double residual{0.0};
};

// Independent implementation of Ledoit-Wolf's QuEST map, defined by
// JMVA 139 (2015), Eqs. (2.11)-(2.17), and numerically evaluated using
// CSDA 115 (2017), Sections 5-10.
[[nodiscard]] QuestForwardResult
quest_forward(std::span<const double> population_eigenvalues,
              std::size_t observations, QuestSpec spec = {});

// Solves the dimension-normalized least-squares problem in JMVA 139 (2015),
// Theorem 2.2. The returned spectrum is positive and sorted.
[[nodiscard]] QuestInverseResult
quest_inverse(std::span<const double> sample_eigenvalues,
              std::size_t observations, QuestSpec spec = {});

// Boundary value associated with the fitted population spectrum. The query
// must lie on the continuous sample support.
[[nodiscard]] QuestStieltjesResult
quest_boundary_stieltjes(std::span<const double> population_eigenvalues,
                         std::size_t observations, double sample_eigenvalue,
                         QuestSpec spec = {});

} // namespace portfolio_math
