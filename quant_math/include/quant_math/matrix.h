#pragma once

#include <cstddef>
#include <span>
#include <string>

#include <Eigen/Core>

namespace quant_math {

using DenseMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using DenseVector = Eigen::VectorXd;

struct MatrixView {
    const double* data{nullptr};
    std::size_t rows{0};
    std::size_t cols{0};
    std::size_t row_stride{0};

    [[nodiscard]] double operator()(std::size_t row, std::size_t col) const noexcept {
        return data[row * row_stride + col];
    }
};

struct MatrixValidation {
    bool ok{false};
    std::string message;
};

[[nodiscard]] MatrixView view(const DenseMatrix& matrix) noexcept;
[[nodiscard]] MatrixValidation validate_finite(MatrixView matrix);
[[nodiscard]] MatrixValidation validate_symmetric(MatrixView matrix, double tolerance);
[[nodiscard]] bool is_positive_semidefinite(
    const DenseMatrix& matrix,
    double tolerance,
    double* minimum_eigenvalue = nullptr);

}  // namespace quant_math
