#include "core/Wind.h"

#include <random>

namespace p2s {

WindField::WindField(const Spec& s, unsigned seed) {
    mean_ = s.vec3("velocity", {0, 0, 0});
    gust_ = s.num("gust", 0.0);
    turbulence_ = s.num("turbulence", 0.0);
    length_ = s.positive("turbulence_length_m", 1.0);
    airDensity_ = s.positive("air_density", 1.2);
    // Random plane-wave modes with amplitude perpendicular to the wave vector: each mode is divergence free.
    std::mt19937 rng(seed);
    std::normal_distribution<double> n(0.0, 1.0);
    for (Mode& m : modes_) {
        const Vec3 dir = Vec3{n(rng), n(rng), n(rng)}.normalized();
        Vec3 a = Vec3{n(rng), n(rng), n(rng)};
        a = (a - dir * a.dot(dir)).normalized();
        m.k = dir * (2 * kPi / length_ * (0.6 + 0.8 * std::uniform_real_distribution<double>(0, 1)(rng)));
        m.amp = a * std::sqrt(2.0 / modes_.size());
        m.phase = std::uniform_real_distribution<double>(0, 2 * kPi)(rng);
    }
}

Vec3 WindField::at(double t, const Vec3& p) const {
    const double speed = mean_.norm();
    const double g = 1.0 + gust_ * (0.55 * std::sin(0.9 * t) + 0.3 * std::sin(2.3 * t + 1.0) + 0.15 * std::sin(5.1 * t + 2.0));
    Vec3 v = mean_ * g;
    if (turbulence_ > 0 && speed > 0) {
        const Vec3 drift = p - mean_ * t;  // eddies are carried by the mean flow
        for (const Mode& m : modes_) v += m.amp * (turbulence_ * speed * std::sin(m.k.dot(drift) + m.phase));
    }
    return v;
}

Json WindField::toJson() const {
    return {{"velocity", p2s::toJson(mean_)}, {"gust", gust_}, {"turbulence", turbulence_},
            {"turbulence_length_m", length_}, {"air_density", airDensity_}};
}

}  // namespace p2s
