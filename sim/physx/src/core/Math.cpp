#include "core/Math.h"

namespace p2s {

Quat Quat::fromAxes(const Vec3& ex, const Vec3& ey, const Vec3& ez) {
    // Shepperd's method on the rotation matrix with columns ex, ey, ez.
    const double m00 = ex.x, m10 = ex.y, m20 = ex.z;
    const double m01 = ey.x, m11 = ey.y, m21 = ey.z;
    const double m02 = ez.x, m12 = ez.y, m22 = ez.z;
    const double tr = m00 + m11 + m22;
    Quat q;
    if (tr > 0) {
        const double s = std::sqrt(tr + 1.0) * 2;
        q = {(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s};
    } else if (m00 > m11 && m00 > m22) {
        const double s = std::sqrt(1.0 + m00 - m11 - m22) * 2;
        q = {0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s};
    } else if (m11 > m22) {
        const double s = std::sqrt(1.0 + m11 - m00 - m22) * 2;
        q = {(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s};
    } else {
        const double s = std::sqrt(1.0 + m22 - m00 - m11) * 2;
        q = {(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s};
    }
    return q.normalized();
}

Quat Quat::between(const Vec3& a, const Vec3& b) {
    const Vec3 u = a.normalized(), v = b.normalized();
    const double d = u.dot(v);
    if (d < -1.0 + 1e-9) {  // opposite: rotate pi about any perpendicular axis
        Vec3 axis = Vec3{1, 0, 0}.cross(u);
        if (axis.norm2() < 1e-9) axis = Vec3{0, 1, 0}.cross(u);
        return axisAngle(axis, kPi);
    }
    const Vec3 c = u.cross(v);
    return Quat{c.x, c.y, c.z, 1.0 + d}.normalized();
}

Quat slerp(const Quat& a, const Quat& bIn, double u) {
    Quat b = bIn;
    double d = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    if (d < 0) { b = {-b.x, -b.y, -b.z, -b.w}; d = -d; }
    if (d > 0.9995) {
        return Quat{a.x + u * (b.x - a.x), a.y + u * (b.y - a.y), a.z + u * (b.z - a.z), a.w + u * (b.w - a.w)}
            .normalized();
    }
    const double th = std::acos(d), s = std::sin(th);
    const double wa = std::sin((1 - u) * th) / s, wb = std::sin(u * th) / s;
    return {wa * a.x + wb * b.x, wa * a.y + wb * b.y, wa * a.z + wb * b.z, wa * a.w + wb * b.w};
}

}  // namespace p2s
