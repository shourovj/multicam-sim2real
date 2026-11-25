#include "world/Events.h"

namespace p2s {

void EventLog::emit(const std::string& name, double t, Json data) {
    events_.push_back({name, t, std::move(data)});
}

std::optional<double> EventLog::first(const std::string& name) const {
    for (const Event& e : events_)
        if (e.name == name) return e.t;
    return std::nullopt;
}

Json EventLog::toJson() const {
    Json out = Json::array();
    for (const Event& e : events_) {
        Json j = {{"name", e.name}, {"t", std::round(e.t * 1e4) / 1e4}};
        if (!e.data.empty()) j["data"] = e.data;
        out.push_back(j);
    }
    return out;
}

Trigger::Trigger(const Spec& s) {
    if (s.has("at_s")) at_ = s.num("at_s");
    onEvent_ = s.str("on_event", "");
    delay_ = s.num("delay_s", 0);
    if (at_ && *at_ < 0) s["at_s"].fail("must be >= 0");
    if (delay_ < 0) s["delay_s"].fail("must be >= 0");
    if (!at_ && onEvent_.empty()) s.fail("a trigger needs at_s or on_event");
    if (at_ && !onEvent_.empty()) s.fail("give either at_s or on_event, not both");
    action_.target = s.str("target");
    action_.verb = s.str("do");
    if (s.has("params")) {  // the target entity validates them when the trigger fires (Entity::act)
        const Spec p = s["params"];
        if (!p.isObject()) p.fail("expected an object");
        action_.params = p.opaque();
    }
}

std::optional<Action> Trigger::poll(double t, const EventLog& events) {
    if (fired_) return std::nullopt;
    double when;
    if (at_) {
        when = *at_;
    } else {
        const auto e = events.first(onEvent_);
        if (!e) return std::nullopt;
        when = *e + delay_;
    }
    if (t + 1e-9 < when) return std::nullopt;
    fired_ = true;
    return action_;
}

std::string Trigger::describe() const {
    const std::string when = at_ ? "at t=" + std::to_string(*at_) : onEvent_ + " + " + std::to_string(delay_) + " s";
    return action_.target + "." + action_.verb + " (" + when + ")";
}

}  // namespace p2s
