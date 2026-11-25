#include "core/Actuators.h"

#include <cmath>
#include <stdexcept>

namespace p2s {

void ActuatorLog::declare(const std::string& id, const std::string& kind, double carriedMassKg, bool torque) {
    ActuatorStats& s = stats_[id];
    s.kind = kind;
    s.carriedMassKg = carriedMassKg;
    s.torque = torque;
}

void ActuatorLog::step(const std::string& id, double force, double power, double dt) {
    const auto it = stats_.find(id);
    if (it == stats_.end()) throw std::logic_error("actuator " + id + " was not declared");
    ActuatorStats& s = it->second;
    s.peakForce = std::max(s.peakForce, std::fabs(force));
    s.peakPower = std::max(s.peakPower, power);
    (power > 0 ? s.workIn : s.workOut) += std::fabs(power) * dt;
    s.lastForce = force;
    s.lastPower = power;
}

double ActuatorLog::netWork() const {
    double w = 0;
    for (const auto& [_, s] : stats_) w += s.workIn - s.workOut;
    return w;
}

double ActuatorLog::totalWorkIn() const {
    double w = 0;
    for (const auto& [_, s] : stats_) w += s.workIn;
    return w;
}

Json ActuatorLog::toJson() const {
    Json out = Json::object();
    for (const auto& [id, s] : stats_)
        out[id] = {{"kind", s.kind},           {"carried_mass_kg", s.carriedMassKg}, {"peak_force", s.peakForce},
                   {"force_unit", s.torque ? "N m" : "N"}, {"peak_power_w", s.peakPower},
                   {"work_in_j", s.workIn},    {"work_out_j", s.workOut}};
    return out;
}

}  // namespace p2s
