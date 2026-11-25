#include "bodies/Signal.h"

namespace p2s {

TrafficSignal::TrafficSignal(const Spec& s, World&) : Entity(s.str("id"), "traffic_signal") {
    for (const Spec& p : s.items("plan")) {
        plan_.push_back({p.oneOf("state", {"red", "yellow", "green"}), p.num("until_s")});
        if (plan_.size() > 1 && plan_.back().until <= plan_[plan_.size() - 2].until) p["until_s"].fail("plan out of order");
    }
    if (plan_.empty()) s.fail("plan needs at least one phase");
    for (const Spec& h : s.items("heads")) heads_.push_back(h.pose());
    last_ = state(0);
}

const std::string& TrafficSignal::state(double t) const {
    for (const Phase& p : plan_)
        if (t < p.until) return p.state;
    return plan_.back().state;
}

void TrafficSignal::afterStep(World& w, double) {
    const std::string& now = state(w.time());
    if (now == last_) return;
    w.events().emit(id() + ":" + now, w.time());
    last_ = now;
}

void TrafficSignal::record(World& w, Recorder& rec) {
    const std::string& s = state(w.time());
    rec.put("signal/" + id(), {s == "red" ? 0.0f : s == "yellow" ? 1.0f : 2.0f});
}

Json TrafficSignal::describe() const {
    Json plan = Json::array(), heads = Json::array();
    for (const Phase& p : plan_) plan.push_back({{"state", p.state}, {"until_s", p.until}});
    for (const Pose& h : heads_) heads.push_back({{"pos", toJson(h.p)}, {"quat_xyzw", toJson(h.q)}});
    return {{"plan", plan}, {"heads", heads}, {"streams", {{"state", "signal/" + id()}}},
            {"stream_columns", {"state (0 red, 1 yellow, 2 green)"}}};
}

}  // namespace p2s
