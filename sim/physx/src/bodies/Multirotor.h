// A multirotor (quad, hexa, octo): a rigid body flown by its rotors. Each rotor gives thrust kf w^2 along the
// body's +Z at its arm position and a reaction torque -spin km w^2 about +Z; motors follow commands with a lag
// and never exceed w_max. A cascaded controller (goal path -> thrust vector -> attitude -> mixer) flies it, so
// every motion comes from rotor forces.
//
//   all rigid-body keys (shapes, pos, mass_kg, drag, ...) plus
//   "rotors": {"layout": "quad_x" | "quad_plus" | "hex" | "octo", "arm_m": 0.25, "height_m": 0.05}
//             or [{"pos": [x, y, z], "spin": 1 | -1}, ...]          (spin +1 = counter-clockwise seen from above)
//   "rotor":  {"thrust_to_weight": 2.0, "max_rpm": 9000, "km_over_kf_m": 0.016, "tau_s": 0.03}
//   "controller": {"goals": [...] (see GoalPath), "yaw_deg": 0, "tilt_max_deg": 35,
//                  "position_bandwidth": 2.2 (rad/s), "attitude_bandwidth": 25 (rad/s)}
// Events: <id>:lift_off. Verbs: rigid-body verbs plus "kill" (motors off) and "goto" {pos, t_s}.
#pragma once

#include "bodies/RigidBody.h"
#include "control/GoalPath.h"
#include "control/Linear.h"

namespace p2s {

class Multirotor : public RigidBody {
public:
    Multirotor(const Spec& s, World& w);

    void link(World& w) override;
    void beforeStep(World& w, double dt) override;
    void afterStep(World& w, double dt) override;
    void record(World& w, Recorder& rec) override;
    Json describe() const override;
    std::vector<std::string> verbs() const override;
    void act(World& w, const Action& a) override;

private:
    struct Rotor {
        Vec3 pos;  // body frame
        double spin;
        double omega = 0;  // rad/s
    };
    struct Wrench {
        double thrust;
        Vec3 torque;  // body frame
    };
    Wrench control(World& w, double dt);
    void mix(const Wrench& want, double dt);
    std::string rotorId(size_t i) const { return id() + "/rotor" + std::to_string(i); }

    std::vector<Rotor> rotors_;
    GoalPath goals_;
    double kf_ = 0, km_ = 0, omegaMax_ = 0, tau_ = 0.03, thrustToWeight_ = 2;
    double yaw_ = 0, tiltMax_ = 0, wPos_ = 2.2, wAtt_ = 25;
    Vec3 home_;
    Matrix mixer_{4, 4};
    bool killed_ = false, onGround_ = false, liftedOff_ = false;
    double startZ_ = 0, lastTilt_ = 0, lastThrust_ = 0;
    Vec3 integral_;
};

}  // namespace p2s
