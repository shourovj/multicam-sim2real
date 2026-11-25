#include "control/Linear.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace p2s {

Matrix Matrix::transpose() const {
    Matrix t(cols_, rows_);
    for (int r = 0; r < rows_; ++r)
        for (int c = 0; c < cols_; ++c) t(c, r) = (*this)(r, c);
    return t;
}

Matrix Matrix::operator*(const Matrix& o) const {
    if (cols_ != o.rows_) throw std::logic_error("matrix size mismatch");
    Matrix m(rows_, o.cols_);
    for (int r = 0; r < rows_; ++r)
        for (int k = 0; k < cols_; ++k)
            for (int c = 0; c < o.cols_; ++c) m(r, c) += (*this)(r, k) * o(k, c);
    return m;
}

std::vector<double> Matrix::operator*(const std::vector<double>& v) const {
    if (int(v.size()) != cols_) throw std::logic_error("matrix-vector size mismatch");
    std::vector<double> out(size_t(rows_), 0.0);
    for (int r = 0; r < rows_; ++r)
        for (int c = 0; c < cols_; ++c) out[size_t(r)] += (*this)(r, c) * v[size_t(c)];
    return out;
}

std::vector<double> Matrix::solve(std::vector<double> b) const {
    if (rows_ != cols_ || int(b.size()) != rows_) throw std::logic_error("solve needs a square system");
    Matrix a = *this;
    const int n = rows_;
    double scale = 0;
    for (double v : a.a_) scale = std::max(scale, std::fabs(v));
    for (int col = 0; col < n; ++col) {
        int piv = col;
        for (int r = col + 1; r < n; ++r)
            if (std::fabs(a(r, col)) > std::fabs(a(piv, col))) piv = r;
        if (std::fabs(a(piv, col)) <= 1e-14 * scale) throw std::runtime_error("singular matrix");
        for (int c = 0; c < n; ++c) std::swap(a(col, c), a(piv, c));
        std::swap(b[size_t(col)], b[size_t(piv)]);
        for (int r = col + 1; r < n; ++r) {
            const double f = a(r, col) / a(col, col);
            for (int c = col; c < n; ++c) a(r, c) -= f * a(col, c);
            b[size_t(r)] -= f * b[size_t(col)];
        }
    }
    std::vector<double> x(size_t(n), 0.0);
    for (int r = n - 1; r >= 0; --r) {
        double s = b[size_t(r)];
        for (int c = r + 1; c < n; ++c) s -= a(r, c) * x[size_t(c)];
        x[size_t(r)] = s / a(r, r);
    }
    return x;
}

Matrix Matrix::pseudoInverse(double lambda) const {
    const Matrix at = transpose();
    Matrix aat = (*this) * at;
    for (int i = 0; i < rows_; ++i) aat(i, i) += lambda * lambda;
    Matrix inv(rows_, rows_);
    for (int c = 0; c < rows_; ++c) {
        std::vector<double> e(size_t(rows_), 0.0);
        e[size_t(c)] = 1.0;
        const auto col = aat.solve(e);
        for (int r = 0; r < rows_; ++r) inv(r, c) = col[size_t(r)];
    }
    return at * inv;
}

}  // namespace p2s
