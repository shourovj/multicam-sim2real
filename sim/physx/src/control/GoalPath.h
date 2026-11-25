// A reference path through timed goals: each goal is reached by a quintic from wherever the body actually is
// when that goal starts (zero acceleration at both ends), as in the legacy sim/physics/quadrotor.py.
//   "goals": [{"from_t": 0.7, "t": 5.1, "pos": [x, y, z], "vel": [0, 0, 0]}, ...]
// from_t defaults to the previous goal's t; vel (on arrival) defaults to zero.
#pragma once

#include <optional>
#include <vector>

#include "core/Spec.h"

namespace p2s {

struct Reference {
    Vec3 pos, vel, acc;
};

class GoalPath {
public:
    GoalPath() = default;
    explicit GoalPath(const Spec& owner, const std::string& key = "goals");

    bool empty() const { return goals_.empty(); }
    bool started(double t) const { return !goals_.empty() && t >= goals_.front().fromT; }
    // Reference at time t; `pos`/`vel` are the body's actual state, used to anchor a goal when it starts.
    Reference at(double t, const Vec3& pos, const Vec3& vel, const Vec3& home);
    Json toJson() const;

private:
    struct Goal {
        double fromT, t;
        Vec3 pos, vel;
        std::optional<Vec3> p0, v0;
    };
    std::vector<Goal> goals_;
};

}  // namespace p2s
