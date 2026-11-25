#include "control/GoalPath.h"

namespace p2s {

GoalPath::GoalPath(const Spec& owner, const std::string& key) {
    double prev = 0;
    for (const Spec& g : owner.items(key)) {
        Goal goal{g.num("from_t", prev), g.num("t"), g.vec3("pos"), g.vec3("vel", {0, 0, 0}), {}, {}};
        if (goal.t <= goal.fromT) g["t"].fail("t must be after from_t");
        if (!goals_.empty() && goal.fromT < goals_.back().fromT) g.fail("goals must be in time order");
        goals_.push_back(goal);
        prev = goal.t;
    }
}

Reference GoalPath::at(double t, const Vec3& pos, const Vec3& vel, const Vec3& home) {
    size_t active = goals_.size();
    for (size_t i = 0; i < goals_.size(); ++i)
        if (goals_[i].fromT <= t) active = i;
    if (active == goals_.size()) return {home, {}, {}};
    Goal& g = goals_[active];
    if (!g.p0) {  // the path starts from the body's actual state, not from where the previous goal ended
        g.p0 = pos;
        g.v0 = vel;
    }
    const double T = g.t - g.fromT, s = t - g.fromT;
    if (s >= T) return {g.pos + g.vel * (s - T), g.vel, {}};
    const Vec3 h = g.pos - *g.p0, v0 = *g.v0, v1 = g.vel;
    const Vec3 a3 = (h * 20 - (v1 * 8 + v0 * 12) * T) / (2 * T * T * T);
    const Vec3 a4 = (h * -30 + (v1 * 14 + v0 * 16) * T) / (2 * T * T * T * T);
    const Vec3 a5 = (h * 12 - (v1 + v0) * 6 * T) / (2 * T * T * T * T * T);
    const double s2 = s * s, s3 = s2 * s, s4 = s3 * s, s5 = s4 * s;
    return {*g.p0 + v0 * s + a3 * s3 + a4 * s4 + a5 * s5, v0 + a3 * (3 * s2) + a4 * (4 * s3) + a5 * (5 * s4),
            a3 * (6 * s) + a4 * (12 * s2) + a5 * (20 * s3)};
}

Json GoalPath::toJson() const {
    Json out = Json::array();
    for (const Goal& g : goals_)
        out.push_back({{"from_t", g.fromT}, {"t", g.t}, {"pos", p2s::toJson(g.pos)}, {"vel", p2s::toJson(g.vel)}});
    return out;
}

}  // namespace p2s
