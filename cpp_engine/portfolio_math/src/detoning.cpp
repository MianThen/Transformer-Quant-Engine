#include "portfolio_math/detoning.h"

#include <algorithm>
#include <cmath>

#include <Eigen/Eigenvalues>

namespace portfolio_math {
namespace {

using quant_math::DenseMatrix;
using quant_math::DenseVector;

DenseMatrix copy_matrix(quant_math::MatrixView source) {
  DenseMatrix result(static_cast<Eigen::Index>(source.rows),
                     static_cast<Eigen::Index>(source.cols));
  for (std::size_t row = 0; row < source.rows; ++row) {
    for (std::size_t col = 0; col < source.cols; ++col) {
      result(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
          source(row, col);
    }
  }
  return result;
}

DetoningResult failed(DetoningStatus status) {
  DetoningResult result;
  result.status = status;
  return result;
}

}  // namespace

bool valid_detoning_spec(const DetoningSpec& spec) noexcept {
  return spec.components <= 1 && std::isfinite(spec.eigenvalue_floor) &&
         spec.eigenvalue_floor > 0.0 && std::isfinite(spec.psd_tolerance) &&
         spec.psd_tolerance > 0.0 &&
         std::isfinite(spec.symmetry_tolerance) &&
         spec.symmetry_tolerance >= 0.0;
}

DetoningResult detone_correlation(quant_math::MatrixView input,
                                  DetoningSpec spec) {
  if (!valid_detoning_spec(spec) || input.rows == 0 ||
      input.rows != input.cols ||
      !quant_math::validate_symmetric(input, spec.symmetry_tolerance).ok) {
    return failed(DetoningStatus::INVALID_INPUT);
  }

  const Eigen::Index dimension = static_cast<Eigen::Index>(input.rows);
  DenseMatrix source = copy_matrix(input);
  for (Eigen::Index index = 0; index < dimension; ++index) {
    if (!std::isfinite(source(index, index)) ||
        source(index, index) <= spec.eigenvalue_floor) {
      return failed(DetoningStatus::INVALID_INPUT);
    }
  }

  DenseMatrix normalized = source;
  const DenseVector diagonal = source.diagonal();
  for (Eigen::Index row = 0; row < dimension; ++row) {
    for (Eigen::Index col = 0; col < dimension; ++col) {
      normalized(row, col) /=
          std::sqrt(diagonal(row) * diagonal(col));
    }
  }
  normalized = 0.5 * (normalized + normalized.transpose()).eval();

  Eigen::SelfAdjointEigenSolver<DenseMatrix> eigensolver(normalized);
  if (eigensolver.info() != Eigen::Success) {
    return failed(DetoningStatus::NUMERICAL_FAILURE);
  }

  DetoningResult result;
  auto& diagnostics = result.diagnostics;
  diagnostics.removed_components = spec.components;
  diagnostics.input_trace = normalized.trace();
  diagnostics.raw_eigenvalues.resize(static_cast<std::size_t>(dimension));
  DenseVector cleaned = eigensolver.eigenvalues();
  for (Eigen::Index index = 0; index < dimension; ++index) {
    const double value = eigensolver.eigenvalues()(index);
    if (!std::isfinite(value) || value < -spec.psd_tolerance) {
      return failed(DetoningStatus::NUMERICAL_FAILURE);
    }
    diagnostics.raw_eigenvalues[static_cast<std::size_t>(index)] =
        std::max(0.0, value);
    cleaned(index) = std::max(0.0, value);
  }
  for (std::uint32_t component = 0; component < spec.components;
       ++component) {
    const Eigen::Index index = dimension - 1 -
                               static_cast<Eigen::Index>(component);
    cleaned(index) = 0.0;
  }

  DenseMatrix detoned = eigensolver.eigenvectors() * cleaned.asDiagonal() *
                        eigensolver.eigenvectors().transpose();
  detoned = 0.5 * (detoned + detoned.transpose()).eval();
  const DenseVector detoned_diagonal = detoned.diagonal();
  if ((detoned_diagonal.array() <= spec.eigenvalue_floor).any()) {
    return failed(DetoningStatus::DEGENERATE_OUTPUT);
  }
  for (Eigen::Index row = 0; row < dimension; ++row) {
    for (Eigen::Index col = 0; col < dimension; ++col) {
      detoned(row, col) /= std::sqrt(detoned_diagonal(row) *
                                     detoned_diagonal(col));
    }
  }
  detoned = 0.5 * (detoned + detoned.transpose()).eval();
  for (Eigen::Index index = 0; index < dimension; ++index) {
    detoned(index, index) = 1.0;
  }
  if (!quant_math::validate_finite(quant_math::view(detoned)).ok ||
      !quant_math::is_positive_semidefinite(detoned, spec.psd_tolerance)) {
    return failed(DetoningStatus::NUMERICAL_FAILURE);
  }

  Eigen::SelfAdjointEigenSolver<DenseMatrix> output_eigensolver(detoned);
  if (output_eigensolver.info() != Eigen::Success) {
    return failed(DetoningStatus::NUMERICAL_FAILURE);
  }
  diagnostics.detoned_eigenvalues.resize(static_cast<std::size_t>(dimension));
  for (Eigen::Index index = 0; index < dimension; ++index) {
    diagnostics.detoned_eigenvalues[static_cast<std::size_t>(index)] =
        std::max(0.0, output_eigensolver.eigenvalues()(index));
  }
  diagnostics.output_trace = detoned.trace();
  diagnostics.trace_drift =
      std::abs(diagnostics.output_trace - diagnostics.input_trace);
  diagnostics.minimum_output_eigenvalue =
      output_eigensolver.eigenvalues().minCoeff();
  diagnostics.psd_repair_amount = 0.0;
  result.correlation = std::move(detoned);
  result.status = DetoningStatus::OK;
  return result;
}

}  // namespace portfolio_math
