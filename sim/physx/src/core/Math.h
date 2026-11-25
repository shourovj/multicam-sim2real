// Small vector, quaternion and pose types shared by the simulator and the checker (no PhysX dependency).
// World frame: Z up, metres, seconds, radians internally.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace p2s {

constexpr double kPi = 3.14159265358979323846;
inline double deg2rad(double d) { return d * kPi / 180.0; }
inline double rad2deg(double r) { return r * 180.0 / kPi; }

struct Vec3 {
    double x = 0, y = 0, z = 0;

    Vec3() = default;
    constexpr Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator-() const { return {-x, -y, -z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3& operator*=(double s) { x *= s; y *= s; z *= s; return *this; }

    double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const { return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x}; }
    double norm2() const { return dot(*this); }
    double norm() const { return std::sqrt(norm2()); }
    Vec3 normalized() const { double n = norm(); return n > 1e-12 ? *this / n : Vec3{0, 0, 0}; }
    Vec3 cwiseMul(const Vec3& o) const { return {x * o.x, y * o.y, z * o.z}; }
    Vec3 cwiseAbs() const { return {std::fabs(x), std::fabs(y), std::fabs(z)}; }
    Vec3 cwiseMax(const Vec3& o) const { return {std::max(x, o.x), std::max(y, o.y), std::max(z, o.z)}; }
    Vec3 cwiseMin(const Vec3& o) const { return {std::min(x, o.x), std::min(y, o.y), std::min(z, o.z)}; }
    double maxCoeff() const { return std::max({x, y, z}); }
    double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
    bool finite() const { return std::isfinite(x) && std::isfinite(y) && std::isfinite(z); }
};
inline Vec3 operator*(double s, const Vec3& v) { return v * s; }

struct Quat {
    double x = 0, y = 0, z = 0, w = 1;

    Quat() = default;
    constexpr Quat(double x_, double y_, double z_, double w_) : x(x_), y(y_), z(z_), w(w_) {}

    static Quat axisAngle(const Vec3& axis, double angle) {
        const Vec3 a = axis.normalized() * std::sin(angle / 2);
        return {a.x, a.y, a.z, std::cos(angle / 2)};
    }
    // Intrinsic Z (yaw), then Y (pitch), then X (roll): R = Rz(yaw) Ry(pitch) Rx(roll).
    static Quat fromYawPitchRoll(double yaw, double pitch, double roll) {
        return axisAngle({0, 0, 1}, yaw) * axisAngle({0, 1, 0}, pitch) * axisAngle({1, 0, 0}, roll);
    }
    // Rotation whose columns are the given orthonormal axes.
    static Quat fromAxes(const Vec3& ex, const Vec3& ey, const Vec3& ez);
    // Shortest rotation taking unit vector a to unit vector b.
    static Quat between(const Vec3& a, const Vec3& b);

    Quat operator*(const Quat& o) const {
        return {w * o.x + x * o.w + y * o.z - z * o.y, w * o.y - x * o.z + y * o.w + z * o.x,
                w * o.z + x * o.y - y * o.x + z * o.w, w * o.w - x * o.x - y * o.y - z * o.z};
    }
    Quat conj() const { return {-x, -y, -z, w}; }
    Quat normalized() const {
        const double n = std::sqrt(x * x + y * y + z * z + w * w);
        return n > 0 ? Quat{x / n, y / n, z / n, w / n} : Quat{};
    }
    Vec3 rotate(const Vec3& v) const {
        const Vec3 u{x, y, z};
        const Vec3 t = u.cross(v) * 2.0;
        return v + t * w + u.cross(t);
    }
    Vec3 rotateInv(const Vec3& v) const { return conj().rotate(v); }
    Vec3 axisX() const { return rotate({1, 0, 0}); }
    Vec3 axisY() const { return rotate({0, 1, 0}); }
    Vec3 axisZ() const { return rotate({0, 0, 1}); }
    double angle() const { return 2.0 * std::acos(std::clamp(std::fabs(w), 0.0, 1.0)); }
    bool finite() const { return std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && std::isfinite(w); }
};

// Spherical linear interpolation along the short arc.
Quat slerp(const Quat& a, const Quat& b, double u);

struct Pose {
    Vec3 p;
    Quat q;

    Vec3 transform(const Vec3& v) const { return q.rotate(v) + p; }
    Vec3 transformInv(const Vec3& v) const { return q.rotateInv(v - p); }
    Pose operator*(const Pose& o) const { return {transform(o.p), q * o.q}; }
    Pose inverse() const { const Quat qi = q.conj(); return {qi.rotate(-p), qi}; }
};

struct Aabb {
    Vec3 lo{1e300, 1e300, 1e300}, hi{-1e300, -1e300, -1e300};

    void add(const Vec3& v) { lo = lo.cwiseMin(v); hi = hi.cwiseMax(v); }
    bool empty() const { return lo.x > hi.x; }
    bool contains(const Vec3& v) const {
        return v.x >= lo.x && v.y >= lo.y && v.z >= lo.z && v.x <= hi.x && v.y <= hi.y && v.z <= hi.z;
    }
    Vec3 center() const { return (lo + hi) * 0.5; }
    Vec3 extent() const { return hi - lo; }
};

}  // namespace p2s
