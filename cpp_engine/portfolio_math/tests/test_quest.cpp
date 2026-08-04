#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "portfolio_math/quest.h"

namespace {

bool check(bool condition, const char *message) {
  if (!condition)
    std::fprintf(stderr, "FAILED: %s\n", message);
  return condition;
}

bool near(double left, double right, double tolerance) {
  return std::abs(left - right) <=
         tolerance * std::max({1.0, std::abs(left), std::abs(right)});
}

bool test_forward_invariants() {
  bool ok = true;
  const std::vector<double> population{1.0, 1.0, 1.0, 1.0};
  const auto first = portfolio_math::quest_forward(population, 8);
  const auto replay = portfolio_math::quest_forward(population, 8);
  ok &= check(first.status == portfolio_math::QuestStatus::OK,
              "equal-spectrum forward status");
  ok &= check(first.sample_eigenvalues.size() == population.size() &&
                  std::is_sorted(first.sample_eigenvalues.begin(),
                                 first.sample_eigenvalues.end()),
              "forward output is sorted and dimension preserving");
  ok &= check(
      first.diagnostics.support.size() == 1 &&
          first.diagnostics.support.front().population_eigenvalue_count == 4,
      "equal population spectrum has one support interval");
  const double root_c = std::sqrt(0.5);
  ok &= check(near(first.diagnostics.support.front().x_left,
                   (1.0 - root_c) * (1.0 - root_c), 2e-9) &&
                  near(first.diagnostics.support.front().x_right,
                       (1.0 + root_c) * (1.0 + root_c), 2e-9),
              "equal-spectrum support matches Marchenko-Pastur endpoints");
  ok &= check(first.sample_eigenvalues == replay.sample_eigenvalues,
              "forward replay is deterministic");

  const auto stieltjes =
      portfolio_math::quest_boundary_stieltjes(population, 8, 1.0);
  ok &= check(stieltjes.status == portfolio_math::QuestStatus::OK &&
                  near(stieltjes.value.real(), -0.5, 2e-9) &&
                  near(stieltjes.value.imag(), std::sqrt(1.75), 2e-9) &&
                  stieltjes.residual < 1e-9,
              "boundary Stieltjes transform matches Marchenko-Pastur oracle");

  std::vector<double> scaled(population.size(), 7.0);
  const auto scaled_result = portfolio_math::quest_forward(scaled, 8);
  ok &= check(scaled_result.status == portfolio_math::QuestStatus::OK,
              "scaled forward status");
  for (std::size_t index = 0; index < population.size(); ++index) {
    ok &= check(near(scaled_result.sample_eigenvalues[index],
                     7.0 * first.sample_eigenvalues[index], 2e-9),
                "QuEST scale equivariance");
  }

  const std::vector<double> separated{1.0, 1.0, 25.0, 25.0};
  const auto split = portfolio_math::quest_forward(separated, 40);
  ok &= check(split.status == portfolio_math::QuestStatus::OK &&
                  split.diagnostics.support.size() == 2,
              "separated population clusters create two supports");
  return ok;
}

bool test_singular_branch_and_guards() {
  bool ok = true;
  const std::vector<double> population{0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
  const auto singular = portfolio_math::quest_forward(population, 4);
  ok &= check(singular.status == portfolio_math::QuestStatus::OK,
              "p greater than n forward status");
  ok &= check(singular.diagnostics.atom_at_zero > 0.0 &&
                  near(singular.sample_eigenvalues[0], 0.0, 1e-12) &&
                  near(singular.sample_eigenvalues[1], 0.0, 1e-12),
              "singular branch returns structural zero eigenvalues");

  const std::vector<double> square{1.0, 2.0, 3.0, 4.0};
  ok &= check(
      portfolio_math::quest_forward(square, 4).status ==
          portfolio_math::QuestStatus::CONCENTRATION_RATIO_TOO_CLOSE_TO_ONE,
      "c equal to one is guarded");
  const std::vector<double> invalid{1.0, 0.0};
  ok &= check(portfolio_math::quest_forward(invalid, 4).status ==
                  portfolio_math::QuestStatus::INVALID_INPUT,
              "non-positive population eigenvalues are rejected");
  return ok;
}

bool test_inverse_projection() {
  bool ok = true;
  const std::vector<double> population{0.7, 1.1, 1.8};
  const auto forward = portfolio_math::quest_forward(population, 12);
  ok &= check(forward.status == portfolio_math::QuestStatus::OK,
              "inverse fixture forward status");
  portfolio_math::QuestSpec spec;
  spec.max_inverse_iterations = 60;
  spec.objective_tolerance = 1e-14;
  const auto inverse =
      portfolio_math::quest_inverse(forward.sample_eigenvalues, 12, spec);
  ok &= check(inverse.status == portfolio_math::QuestStatus::OK,
              "inverse projection converges");
  ok &= check(inverse.population_eigenvalues.size() == population.size() &&
                  std::is_sorted(inverse.population_eigenvalues.begin(),
                                 inverse.population_eigenvalues.end()),
              "inverse spectrum is positive and sorted");
  ok &= check(std::all_of(inverse.population_eigenvalues.begin(),
                          inverse.population_eigenvalues.end(),
                          [](double value) { return value > 0.0; }),
              "inverse spectrum is strictly positive");
  ok &= check(inverse.diagnostics.maximum_forward_residual < 1e-5,
              "inverse satisfies Theorem 2.2 forward objective");
  return ok;
}

} // namespace

int main() {
  if (!(test_forward_invariants() && test_singular_branch_and_guards() &&
        test_inverse_projection()))
    return 1;
  std::printf("test_quest: all checks passed\n");
  return 0;
}
