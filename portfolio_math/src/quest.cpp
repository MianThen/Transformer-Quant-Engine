#include "portfolio_math/quest.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Core>

namespace portfolio_math {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

struct Atom {
  double value{0.0};
  std::size_t count{0};
};

struct Branch {
  double u_left{0.0};
  double u_right{0.0};
  std::size_t count{0};
};

struct CdfNode {
  double probability{0.0};
  double value{0.0};
};

bool valid_spec(const QuestSpec &spec) {
  return spec.max_root_iterations > 0 && spec.max_inverse_iterations > 0 &&
         spec.max_line_search_steps > 0 && std::isfinite(spec.root_tolerance) &&
         spec.root_tolerance > 0.0 && std::isfinite(spec.objective_tolerance) &&
         spec.objective_tolerance >= 0.0 &&
         std::isfinite(spec.step_tolerance) && spec.step_tolerance > 0.0 &&
         std::isfinite(spec.finite_difference_step) &&
         spec.finite_difference_step > 0.0 &&
         std::isfinite(spec.initial_damping) && spec.initial_damping > 0.0 &&
         std::isfinite(spec.concentration_ratio_guard) &&
         spec.concentration_ratio_guard >= 0.0 &&
         std::isfinite(spec.eigenvalue_floor) && spec.eigenvalue_floor > 0.0 &&
         std::isfinite(spec.angle_row_mass_tolerance) &&
         spec.angle_row_mass_tolerance >= 0.0;
}

template <class Function>
bool bisect(Function function, double left, double right, double tolerance,
            std::uint32_t max_iterations, double &root) {
  double f_left = function(left);
  double f_right = function(right);
  if (!std::isfinite(f_left) || !std::isfinite(f_right) ||
      (f_left > 0.0) == (f_right > 0.0)) {
    return false;
  }
  for (std::uint32_t iteration = 0; iteration < max_iterations; ++iteration) {
    const double middle = std::midpoint(left, right);
    const double f_middle = function(middle);
    if (!std::isfinite(f_middle))
      return false;
    if (std::abs(f_middle) <= tolerance ||
        right - left <=
            tolerance * std::max({1.0, std::abs(left), std::abs(right)})) {
      root = middle;
      return true;
    }
    if ((f_left > 0.0) == (f_middle > 0.0)) {
      left = middle;
      f_left = f_middle;
    } else {
      right = middle;
      f_right = f_middle;
    }
  }
  root = std::midpoint(left, right);
  return std::isfinite(root);
}

std::vector<Atom> group_spectrum(const std::vector<double> &sorted) {
  std::vector<Atom> atoms;
  for (double value : sorted) {
    if (atoms.empty() || value != atoms.back().value) {
      atoms.push_back({value, 1});
    } else {
      ++atoms.back().count;
    }
  }
  return atoms;
}

double phi(const std::vector<Atom> &atoms, std::size_t dimension, double u) {
  double result = 0.0;
  for (const auto &atom : atoms) {
    const double delta = atom.value - u;
    const double weight =
        static_cast<double>(atom.count) / static_cast<double>(dimension);
    result += weight * atom.value * atom.value / (delta * delta);
  }
  return result;
}

double phi_derivative(const std::vector<Atom> &atoms, std::size_t dimension,
                      double u) {
  double result = 0.0;
  for (const auto &atom : atoms) {
    const double delta = atom.value - u;
    const double weight =
        static_cast<double>(atom.count) / static_cast<double>(dimension);
    result += 2.0 * weight * atom.value * atom.value / (delta * delta * delta);
  }
  return result;
}

bool find_support(const std::vector<Atom> &atoms, std::size_t dimension,
                  double c, const QuestSpec &spec,
                  std::vector<Branch> &branches) {
  const double target = 1.0 / c;
  const double squared_mean =
      std::accumulate(atoms.begin(), atoms.end(), 0.0,
                      [dimension](double sum, const Atom &atom) {
                        return sum + static_cast<double>(atom.count) *
                                         atom.value * atom.value /
                                         static_cast<double>(dimension);
                      });
  const double span = std::sqrt(c * squared_mean) + 1.0;
  double outer_left = 0.0;
  double outer_right = 0.0;
  const auto endpoint = [&](double u) {
    return phi(atoms, dimension, u) - target;
  };
  const double first = atoms.front().value;
  const double last = atoms.back().value;
  const double near_left =
      first - 0.5 *
                  std::sqrt(c * static_cast<double>(atoms.front().count) /
                            static_cast<double>(dimension)) *
                  first;
  const double near_right =
      last + 0.5 *
                 std::sqrt(c * static_cast<double>(atoms.back().count) /
                           static_cast<double>(dimension)) *
                 last;
  if (!bisect(endpoint, first - span, near_left, spec.root_tolerance,
              spec.max_root_iterations, outer_left) ||
      !bisect(endpoint, near_right, last + span, spec.root_tolerance,
              spec.max_root_iterations, outer_right)) {
    return false;
  }

  std::vector<std::pair<double, double>> gaps;
  std::vector<std::size_t> split_after;
  for (std::size_t index = 0; index + 1 < atoms.size(); ++index) {
    const double left_atom = atoms[index].value;
    const double right_atom = atoms[index + 1].value;
    const double epsilon = 32.0 * std::numeric_limits<double>::epsilon();
    const double left =
        left_atom + epsilon * std::max(1.0, std::abs(left_atom));
    const double right =
        right_atom - epsilon * std::max(1.0, std::abs(right_atom));
    double minimizer = 0.0;
    if (left >= right ||
        !bisect([&](double u) { return phi_derivative(atoms, dimension, u); },
                left, right, spec.root_tolerance, spec.max_root_iterations,
                minimizer)) {
      return false;
    }
    if (phi(atoms, dimension, minimizer) >= target)
      continue;
    double gap_left = 0.0;
    double gap_right = 0.0;
    if (!bisect(endpoint, left, minimizer, spec.root_tolerance,
                spec.max_root_iterations, gap_left) ||
        !bisect(endpoint, minimizer, right, spec.root_tolerance,
                spec.max_root_iterations, gap_right)) {
      return false;
    }
    gaps.emplace_back(gap_left, gap_right);
    split_after.push_back(index);
  }

  double branch_left = outer_left;
  std::size_t first_atom_index = 0;
  for (std::size_t gap = 0; gap < gaps.size(); ++gap) {
    std::size_t count = 0;
    for (std::size_t index = first_atom_index; index <= split_after[gap];
         ++index) {
      count += atoms[index].count;
    }
    branches.push_back({branch_left, gaps[gap].first, count});
    branch_left = gaps[gap].second;
    first_atom_index = split_after[gap] + 1;
  }
  std::size_t count = 0;
  for (std::size_t index = first_atom_index; index < atoms.size(); ++index) {
    count += atoms[index].count;
  }
  branches.push_back({branch_left, outer_right, count});
  return true;
}

std::complex<double> delta(const std::vector<double> &population, double c,
                           std::complex<double> z) {
  std::complex<double> mean{0.0, 0.0};
  for (double value : population)
    mean += value / (value - z);
  mean /= static_cast<double>(population.size());
  return z - c * z * mean;
}

double map_to_sample(const std::vector<double> &population, double c,
                     std::complex<double> z) {
  return delta(population, c, z).real();
}

bool solve_imaginary_part(const std::vector<double> &population, double c,
                          double u, const QuestSpec &spec, double &v) {
  const auto gamma = [&](double candidate) {
    double sum = 0.0;
    for (double value : population) {
      const double difference = value - u;
      sum += value * value / (difference * difference + candidate * candidate);
    }
    return sum / static_cast<double>(population.size()) - 1.0 / c;
  };
  const double at_zero = gamma(0.0);
  if (std::abs(at_zero) <= spec.root_tolerance) {
    v = 0.0;
    return true;
  }
  if (!(at_zero > 0.0))
    return false;
  const double upper =
      std::sqrt(c *
                std::inner_product(population.begin(), population.end(),
                                   population.begin(), 0.0) /
                static_cast<double>(population.size())) +
      1.0;
  return bisect(gamma, 0.0, upper, spec.root_tolerance,
                spec.max_root_iterations, v);
}

double integrate_inverse(const std::vector<CdfNode> &nodes,
                         double probability) {
  if (probability <= 0.0)
    return 0.0;
  double integral = 0.0;
  for (std::size_t index = 1; index < nodes.size(); ++index) {
    const double p0 = nodes[index - 1].probability;
    const double p1 = nodes[index].probability;
    if (probability <= p0)
      break;
    const double upper = std::min(probability, p1);
    if (upper <= p0 || p1 <= p0)
      continue;
    const double fraction = (upper - p0) / (p1 - p0);
    const double x_upper =
        nodes[index - 1].value +
        fraction * (nodes[index].value - nodes[index - 1].value);
    integral += (upper - p0) * (nodes[index - 1].value + x_upper) * 0.5;
    if (probability <= p1)
      break;
  }
  return integral;
}

double mse(std::span<const double> left, std::span<const double> right) {
  double value = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    const double difference = left[index] - right[index];
    value += difference * difference;
  }
  return value / static_cast<double>(left.size());
}

} // namespace

QuestForwardResult quest_forward(std::span<const double> population_eigenvalues,
                                 std::size_t observations, QuestSpec spec) {
  QuestForwardResult result;
  const std::size_t dimension = population_eigenvalues.size();
  if (dimension == 0 || observations == 0 || !valid_spec(spec))
    return result;
  std::vector<double> population(population_eigenvalues.begin(),
                                 population_eigenvalues.end());
  for (double value : population) {
    if (!std::isfinite(value) || value < spec.eigenvalue_floor)
      return result;
  }
  std::sort(population.begin(), population.end());
  const double c =
      static_cast<double>(dimension) / static_cast<double>(observations);
  result.diagnostics.concentration_ratio = c;
  if (std::abs(c - 1.0) <= spec.concentration_ratio_guard) {
    result.status = QuestStatus::CONCENTRATION_RATIO_TOO_CLOSE_TO_ONE;
    return result;
  }

  const auto atoms = group_spectrum(population);
  std::vector<Branch> branches;
  if (!find_support(atoms, dimension, c, spec, branches)) {
    result.status = QuestStatus::NUMERICAL_FAILURE;
    return result;
  }

  const double atom_at_zero = std::max(0.0, 1.0 - 1.0 / c);
  const double continuous_mass = 1.0 - atom_at_zero;
  result.diagnostics.atom_at_zero = atom_at_zero;
  std::vector<CdfNode> inverse_cdf{{0.0, 0.0}};
  if (atom_at_zero > 0.0)
    inverse_cdf.push_back({atom_at_zero, 0.0});
  double branch_probability = atom_at_zero;

  for (const auto &branch : branches) {
    const std::size_t points = branch.count + 2;
    std::vector<double> x(points, 0.0);
    std::vector<double> density(points, 0.0);
    for (std::size_t index = 0; index < points; ++index) {
      const double angle = kPi * static_cast<double>(index) /
                           (2.0 * static_cast<double>(branch.count + 1));
      const double sine = std::sin(angle);
      const double u =
          branch.u_left + (branch.u_right - branch.u_left) * sine * sine;
      double v = 0.0;
      if (index != 0 && index + 1 != points) {
        if (!solve_imaginary_part(population, c, u, spec, v)) {
          result.status = QuestStatus::NUMERICAL_FAILURE;
          return result;
        }
      }
      const std::complex<double> z{u, v};
      x[index] = map_to_sample(population, c, z);
      density[index] = v / (c * kPi * (u * u + v * v));
      if (!std::isfinite(x[index]) || !std::isfinite(density[index])) {
        result.status = QuestStatus::NUMERICAL_FAILURE;
        return result;
      }
    }
    for (std::size_t index = 1; index < points; ++index) {
      if (x[index] < x[index - 1]) {
        result.status = QuestStatus::NUMERICAL_FAILURE;
        return result;
      }
    }
    const double desired_mass = continuous_mass *
                                static_cast<double>(branch.count) /
                                static_cast<double>(dimension);
    std::vector<double> raw_cdf(points, 0.0);
    for (std::size_t index = 1; index < points; ++index) {
      raw_cdf[index] =
          raw_cdf[index - 1] + 0.5 * (x[index] - x[index - 1]) *
                                   (density[index] + density[index - 1]);
    }
    const double raw_mass = raw_cdf.back();
    if (!(raw_mass > 0.0) || !std::isfinite(raw_mass)) {
      result.status = QuestStatus::NUMERICAL_FAILURE;
      return result;
    }
    result.diagnostics.maximum_raw_mass_error =
        std::max(result.diagnostics.maximum_raw_mass_error,
                 std::abs(raw_mass - desired_mass));
    for (std::size_t index = 0; index < points; ++index) {
      const double probability =
          branch_probability + desired_mass * raw_cdf[index] / raw_mass;
      if (inverse_cdf.empty() || probability > inverse_cdf.back().probability) {
        inverse_cdf.push_back({probability, std::max(0.0, x[index])});
      }
    }
    result.diagnostics.support.push_back(
        {branch.u_left, branch.u_right, std::max(0.0, x.front()),
         std::max(0.0, x.back()), branch.count});
    result.diagnostics.grid_size += points;
    branch_probability += desired_mass;
  }
  if (inverse_cdf.empty() || std::abs(branch_probability - 1.0) > 1e-10) {
    result.status = QuestStatus::NUMERICAL_FAILURE;
    return result;
  }
  inverse_cdf.back().probability = 1.0;
  result.sample_eigenvalues.resize(dimension);
  double previous_integral = 0.0;
  for (std::size_t index = 0; index < dimension; ++index) {
    const double probability =
        static_cast<double>(index + 1) / static_cast<double>(dimension);
    const double integral = integrate_inverse(inverse_cdf, probability);
    result.sample_eigenvalues[index] =
        static_cast<double>(dimension) * (integral - previous_integral);
    previous_integral = integral;
  }
  result.status = QuestStatus::OK;
  return result;
}

QuestInverseResult quest_inverse(std::span<const double> sample_eigenvalues,
                                 std::size_t observations, QuestSpec spec) {
  QuestInverseResult result;
  const std::size_t dimension = sample_eigenvalues.size();
  if (dimension == 0 || observations == 0 || !valid_spec(spec))
    return result;
  std::vector<double> sample(sample_eigenvalues.begin(),
                             sample_eigenvalues.end());
  for (double value : sample) {
    if (!std::isfinite(value) || value < 0.0)
      return result;
  }
  std::sort(sample.begin(), sample.end());
  const double c =
      static_cast<double>(dimension) / static_cast<double>(observations);
  result.diagnostics.concentration_ratio = c;
  if (std::abs(c - 1.0) <= spec.concentration_ratio_guard) {
    result.status = QuestStatus::CONCENTRATION_RATIO_TOO_CLOSE_TO_ONE;
    return result;
  }
  if (dimension == 1) {
    if (!(sample.front() > 0.0))
      return result;
    const std::vector<double> unit_population{1.0};
    const auto unit = quest_forward(unit_population, observations, spec);
    if (unit.status != QuestStatus::OK ||
        !(unit.sample_eigenvalues.front() > 0.0)) {
      result.status = QuestStatus::NUMERICAL_FAILURE;
      return result;
    }
    result.population_eigenvalues = {sample.front() /
                                     unit.sample_eigenvalues.front()};
    const auto replay =
        quest_forward(result.population_eigenvalues, observations, spec);
    if (replay.status != QuestStatus::OK) {
      result.status = replay.status;
      return result;
    }
    result.diagnostics = replay.diagnostics;
    result.diagnostics.iterations = 1;
    result.diagnostics.maximum_forward_residual =
        std::abs(replay.sample_eigenvalues.front() - sample.front());
    result.diagnostics.objective = result.diagnostics.maximum_forward_residual *
                                   result.diagnostics.maximum_forward_residual;
    result.status = QuestStatus::OK;
    return result;
  }
  const std::size_t structural_zeros =
      dimension > observations ? dimension - observations : 0;
  const double positive_sum =
      std::accumulate(sample.begin() + structural_zeros, sample.end(), 0.0);
  const std::size_t positive_count = dimension - structural_zeros;
  if (positive_count == 0 || positive_sum <= 0.0)
    return result;
  const double mean = positive_sum / static_cast<double>(positive_count);

  Eigen::VectorXd parameters(static_cast<Eigen::Index>(dimension));
  for (std::size_t index = 0; index < dimension; ++index) {
    double initial = mean;
    if (index >= structural_zeros && sample[index] > 0.0)
      initial = sample[index];
    parameters(static_cast<Eigen::Index>(index)) =
        std::log(std::max(initial, spec.eigenvalue_floor));
  }

  auto evaluate = [&](const Eigen::VectorXd &candidate,
                      std::vector<double> &population,
                      QuestForwardResult &forward) {
    population.resize(dimension);
    for (std::size_t index = 0; index < dimension; ++index) {
      population[index] = std::max(
          spec.eigenvalue_floor,
          std::exp(std::clamp(candidate(static_cast<Eigen::Index>(index)),
                              -60.0, 60.0)));
    }
    std::sort(population.begin(), population.end());
    forward = quest_forward(population, observations, spec);
    return forward.status == QuestStatus::OK;
  };

  std::vector<double> population;
  QuestForwardResult forward;
  if (!evaluate(parameters, population, forward)) {
    result.status = forward.status;
    return result;
  }
  double objective = mse(forward.sample_eigenvalues, sample);
  double damping = spec.initial_damping;
  QuestStatus terminal_status = QuestStatus::MAX_ITERATIONS;
  std::uint32_t completed_iterations = 0;
  for (std::uint32_t iteration = 0; iteration < spec.max_inverse_iterations;
       ++iteration) {
    completed_iterations = iteration + 1;
    if (objective <= spec.objective_tolerance *
                         std::max(spec.eigenvalue_floor * spec.eigenvalue_floor,
                                  mean * mean)) {
      terminal_status = QuestStatus::OK;
      break;
    }
    Eigen::MatrixXd jacobian(static_cast<Eigen::Index>(dimension),
                             static_cast<Eigen::Index>(dimension));
    bool jacobian_ok = true;
    for (std::size_t column = 0; column < dimension; ++column) {
      Eigen::VectorXd shifted = parameters;
      shifted(static_cast<Eigen::Index>(column)) += spec.finite_difference_step;
      std::vector<double> shifted_population;
      QuestForwardResult shifted_forward;
      if (!evaluate(shifted, shifted_population, shifted_forward)) {
        jacobian_ok = false;
        break;
      }
      for (std::size_t row = 0; row < dimension; ++row) {
        jacobian(static_cast<Eigen::Index>(row),
                 static_cast<Eigen::Index>(column)) =
            (shifted_forward.sample_eigenvalues[row] -
             forward.sample_eigenvalues[row]) /
            spec.finite_difference_step;
      }
    }
    if (!jacobian_ok) {
      terminal_status = QuestStatus::NUMERICAL_FAILURE;
      break;
    }
    Eigen::VectorXd residual(static_cast<Eigen::Index>(dimension));
    for (std::size_t index = 0; index < dimension; ++index) {
      residual(static_cast<Eigen::Index>(index)) =
          forward.sample_eigenvalues[index] - sample[index];
    }
    const Eigen::VectorXd gradient =
        jacobian.transpose() * residual / static_cast<double>(dimension);
    const double parameter_scale =
        std::max(spec.eigenvalue_floor * spec.eigenvalue_floor, mean * mean);
    if (gradient.lpNorm<Eigen::Infinity>() <=
        spec.step_tolerance * parameter_scale) {
      terminal_status = QuestStatus::OK;
      break;
    }
    Eigen::MatrixXd normal = jacobian.transpose() * jacobian;
    normal.diagonal().array() += damping;
    const Eigen::VectorXd step =
        normal.ldlt().solve(-jacobian.transpose() * residual);
    if (!step.allFinite()) {
      terminal_status = QuestStatus::NUMERICAL_FAILURE;
      break;
    }
    if (step.lpNorm<Eigen::Infinity>() <= spec.step_tolerance) {
      terminal_status = QuestStatus::OK;
      break;
    }
    bool accepted = false;
    for (std::uint32_t line = 0; line < spec.max_line_search_steps; ++line) {
      const double scale = std::ldexp(1.0, -static_cast<int>(line));
      const Eigen::VectorXd candidate = parameters + scale * step;
      std::vector<double> candidate_population;
      QuestForwardResult candidate_forward;
      if (!evaluate(candidate, candidate_population, candidate_forward))
        continue;
      const double candidate_objective =
          mse(candidate_forward.sample_eigenvalues, sample);
      if (candidate_objective < objective) {
        parameters = candidate;
        population = std::move(candidate_population);
        forward = std::move(candidate_forward);
        objective = candidate_objective;
        damping = std::max(1e-12, damping * 0.3);
        accepted = true;
        break;
      }
    }
    if (!accepted) {
      const double smallest_line_step =
          std::ldexp(step.lpNorm<Eigen::Infinity>(),
                     -static_cast<int>(spec.max_line_search_steps - 1));
      if (smallest_line_step <= spec.step_tolerance) {
        terminal_status = QuestStatus::OK;
        break;
      }
      damping *= 10.0;
      if (!std::isfinite(damping) || damping > 1e16)
        break;
    }
  }
  result.status = terminal_status;
  result.population_eigenvalues = std::move(population);
  result.diagnostics = std::move(forward.diagnostics);
  result.diagnostics.iterations = completed_iterations;
  result.diagnostics.objective = objective;
  if (!forward.sample_eigenvalues.empty()) {
    for (std::size_t index = 0; index < dimension; ++index) {
      result.diagnostics.maximum_forward_residual =
          std::max(result.diagnostics.maximum_forward_residual,
                   std::abs(forward.sample_eigenvalues[index] - sample[index]));
    }
  }
  return result;
}

QuestStieltjesResult
quest_boundary_stieltjes(std::span<const double> population_eigenvalues,
                         std::size_t observations, double sample_eigenvalue,
                         QuestSpec spec) {
  QuestStieltjesResult result;
  const std::size_t dimension = population_eigenvalues.size();
  if (dimension == 0 || observations == 0 || !valid_spec(spec) ||
      !std::isfinite(sample_eigenvalue) || sample_eigenvalue <= 0.0) {
    return result;
  }
  std::vector<double> population(population_eigenvalues.begin(),
                                 population_eigenvalues.end());
  for (double value : population) {
    if (!std::isfinite(value) || value < spec.eigenvalue_floor)
      return result;
  }
  std::sort(population.begin(), population.end());
  const double c =
      static_cast<double>(dimension) / static_cast<double>(observations);
  if (std::abs(c - 1.0) <= spec.concentration_ratio_guard) {
    result.status = QuestStatus::CONCENTRATION_RATIO_TOO_CLOSE_TO_ONE;
    return result;
  }
  const auto atoms = group_spectrum(population);
  if (atoms.size() == 1) {
    const double tau = atoms.front().value;
    const double root_c = std::sqrt(c);
    const double left = tau * (1.0 - root_c) * (1.0 - root_c);
    const double right = tau * (1.0 + root_c) * (1.0 + root_c);
    const double tolerance = 64.0 * spec.root_tolerance *
                             std::max({1.0, std::abs(left), std::abs(right)});
    if (sample_eigenvalue < left - tolerance ||
        sample_eigenvalue > right + tolerance) {
      result.status = QuestStatus::NUMERICAL_FAILURE;
      return result;
    }
    const double radicand =
        std::max(0.0, (right - sample_eigenvalue) * (sample_eigenvalue - left));
    result.value = {tau * (1.0 - c) - sample_eigenvalue, std::sqrt(radicand)};
    result.value /= 2.0 * c * sample_eigenvalue * tau;
    result.mapped_sample_eigenvalue = sample_eigenvalue;
    result.residual = 0.0;
    result.status = QuestStatus::OK;
    return result;
  }

  std::vector<Branch> branches;
  if (!find_support(atoms, dimension, c, spec, branches)) {
    result.status = QuestStatus::NUMERICAL_FAILURE;
    return result;
  }

  for (const auto &branch : branches) {
    const double x_left = map_to_sample(population, c, {branch.u_left, 0.0});
    const double x_right = map_to_sample(population, c, {branch.u_right, 0.0});
    const double support_tolerance =
        64.0 * spec.root_tolerance *
        std::max({1.0, std::abs(x_left), std::abs(x_right)});
    if (sample_eigenvalue < x_left - support_tolerance ||
        sample_eigenvalue > x_right + support_tolerance) {
      continue;
    }

    double u = branch.u_left;
    if (std::abs(sample_eigenvalue - x_left) > support_tolerance &&
        std::abs(sample_eigenvalue - x_right) > support_tolerance) {
      const auto mapped_residual = [&](double candidate_u) {
        if (candidate_u == branch.u_left)
          return x_left - sample_eigenvalue;
        if (candidate_u == branch.u_right)
          return x_right - sample_eigenvalue;
        double candidate_v = 0.0;
        if (!solve_imaginary_part(population, c, candidate_u, spec,
                                  candidate_v)) {
          return std::numeric_limits<double>::quiet_NaN();
        }
        return map_to_sample(population, c, {candidate_u, candidate_v}) -
               sample_eigenvalue;
      };
      if (!bisect(mapped_residual, branch.u_left, branch.u_right,
                  spec.root_tolerance, spec.max_root_iterations, u)) {
        result.status = QuestStatus::NUMERICAL_FAILURE;
        return result;
      }
    } else if (std::abs(sample_eigenvalue - x_right) <= support_tolerance) {
      u = branch.u_right;
    }

    double v = 0.0;
    if (!solve_imaginary_part(population, c, u, spec, v)) {
      result.status = QuestStatus::NUMERICAL_FAILURE;
      return result;
    }
    const std::complex<double> z{u, v};
    const std::complex<double> mapped = delta(population, c, z);
    const std::complex<double> companion = -1.0 / z;
    result.value = (companion + (1.0 - c) / sample_eigenvalue) / c;
    result.mapped_sample_eigenvalue = mapped.real();
    result.residual =
        std::hypot(mapped.real() - sample_eigenvalue, mapped.imag());
    if (!std::isfinite(result.value.real()) ||
        !std::isfinite(result.value.imag()) ||
        !std::isfinite(result.residual)) {
      result.status = QuestStatus::NUMERICAL_FAILURE;
      return result;
    }
    result.status = QuestStatus::OK;
    return result;
  }

  result.status = QuestStatus::NUMERICAL_FAILURE;
  return result;
}

} // namespace portfolio_math
