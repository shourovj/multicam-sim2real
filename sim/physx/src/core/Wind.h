// Air velocity field. world.wind = {
//   "velocity": [vx, vy, vz],          mean wind, m/s
//   "gust": 0.2,                        relative amplitude of slow gusts (three incommensurate sines)
//   "turbulence": 0.5,                  RMS of the turbulent velocity, as a fraction of the mean speed
//   "turbulence_length_m": 1.0,         size of the eddies; they drift with the mean wind (frozen turbulence)
//   "air_density": 1.2 }
#pragma once

#include <array>

#include "core/Spec.h"

namespace p2s {

class WindField {
public:
    WindField() = default;
    explicit WindField(const Spec& s, unsigned seed);

    Vec3 at(double t, const Vec3& p) const;
    double airDensity() const { return airDensity_; }
    Vec3 mean() const { return mean_; }
    Json toJson() const;

private:
    struct Mode {
        Vec3 k, amp;
        double phase;
    };
    Vec3 mean_;
    double gust_ = 0, turbulence_ = 0, length_ = 1, airDensity_ = 1.2;
    std::array<Mode, 8> modes_{};
};

}  // namespace p2s
