// Tiny dense linear algebra for controllers (mixers, IK): row-major matrices of doubles.
#pragma once

#include <vector>

namespace p2s {

class Matrix {
public:
    Matrix(int rows, int cols) : rows_(rows), cols_(cols), a_(size_t(rows * cols), 0.0) {}
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    double& operator()(int r, int c) { return a_[size_t(r * cols_ + c)]; }
    double operator()(int r, int c) const { return a_[size_t(r * cols_ + c)]; }

    Matrix transpose() const;
    Matrix operator*(const Matrix& o) const;
    std::vector<double> operator*(const std::vector<double>& v) const;
    // Solves A x = b for square A (Gaussian elimination, partial pivoting). Throws if singular.
    std::vector<double> solve(std::vector<double> b) const;
    // Damped least-squares pseudo-inverse: A^T (A A^T + lambda^2 I)^-1.
    Matrix pseudoInverse(double lambda = 0.0) const;

private:
    int rows_, cols_;
    std::vector<double> a_;
};

}  // namespace p2s
