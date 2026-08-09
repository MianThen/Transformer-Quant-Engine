#include "quant_math/matrix.h"

#include <algorithm>
#include <cmath>

#include <Eigen/Eigenvalues>

namespace quant_math {

MatrixView view(const DenseMatrix& matrix) noexcept {
    return {matrix.data(), static_cast<std::size_t>(matrix.rows()),
            static_cast<std::size_t>(matrix.cols()), static_cast<std::size_t>(matrix.cols())};
}

MatrixValidation validate_finite(MatrixView matrix) {
    if (matrix.rows != 0 && (matrix.data == nullptr || matrix.row_stride < matrix.cols)) {
        return {false, "invalid matrix storage"};
    }
    for (std::size_t row = 0; row < matrix.rows; ++row) {
        for (std::size_t col = 0; col < matrix.cols; ++col) {
            if (!std::isfinite(matrix(row, col))) {
                return {false, "matrix contains NaN or infinity"};
            }
        }
    }
    return {true, {}};
}

MatrixValidation validate_symmetric(MatrixView matrix, double tolerance) {
    if (tolerance < 0.0 || !std::isfinite(tolerance)) {
        return {false, "invalid symmetry tolerance"};
    }
    const auto finite = validate_finite(matrix);
    if (!finite.ok) return finite;
    if (matrix.rows != matrix.cols) return {false, "matrix is not square"};
    for (std::size_t row = 0; row < matrix.rows; ++row) {
        for (std::size_t col = row + 1; col < matrix.cols; ++col) {
            const double scale = std::max({1.0, std::abs(matrix(row, col)),
                                           std::abs(matrix(col, row))});
            if (std::abs(matrix(row, col) - matrix(col, row)) > tolerance * scale) {
                return {false, "matrix is not symmetric within tolerance"};
            }
        }
    }
    return {true, {}};
}

bool is_positive_semidefinite(
    const DenseMatrix& matrix, double tolerance, double* minimum_eigenvalue) {
    if (!validate_symmetric(view(matrix), tolerance).ok || matrix.rows() == 0) return false;
    const DenseMatrix symmetric = 0.5 * (matrix + matrix.transpose());
    Eigen::SelfAdjointEigenSolver<DenseMatrix> solver(symmetric, Eigen::EigenvaluesOnly);
    if (solver.info() != Eigen::Success) return false;
    const double minimum = solver.eigenvalues().minCoeff();
    if (minimum_eigenvalue != nullptr) *minimum_eigenvalue = minimum;
    const double scale = std::max(1.0, symmetric.diagonal().cwiseAbs().maxCoeff());
    return minimum >= -tolerance * scale;
}

}  // namespace quant_math
