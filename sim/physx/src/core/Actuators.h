// What every actuator (rotor, wheel drive, brake, joint motor, traction controller) did: peak force and power,
// and the net work it put into the world, which bounds how much the world's energy may grow.
#pragma once

#include <map>
#include <string>

#include "core/Spec.h"

namespace p2s {

struct ActuatorStats {
    std::string kind;          // "rotor", "wheel_drive", "brake", "joint_motor", "traction", "thruster", ...
    double carriedMassKg = 0;  // mass the actuator moves (for thrust-to-weight, power-to-mass)
    double peakForce = 0;      // N (or N m for torque actuators, see `torque`)
    double peakPower = 0;      // W, positive = into the world
    double workIn = 0;         // J, integral of positive power
    double workOut = 0;        // J, integral of negative power (braking), as a positive number
    bool torque = false;
    double lastForce = 0, lastPower = 0;
};

class ActuatorLog {
public:
    void declare(const std::string& id, const std::string& kind, double carriedMassKg, bool torque = false);
    // Called every physics step with the force (or torque) applied and the mechanical power delivered.
    void step(const std::string& id, double force, double power, double dt);
    double netWork() const;  // J, sum over actuators of (workIn - workOut)
    double totalWorkIn() const;
    const std::map<std::string, ActuatorStats>& all() const { return stats_; }
    Json toJson() const;

private:
    std::map<std::string, ActuatorStats> stats_;
};

}  // namespace p2s
