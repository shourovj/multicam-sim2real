// Joints between two rigid entities (or one entity and the world).
//
//   {"id": "hinge", "type": "revolute", "a": "frame" | "world", "b": "door",
//    "anchor": [x, y, z] (world, at t = 0), "axis": [0, 0, 1] (world; revolute/prismatic),
//    "limits": {"lower": -90, "upper": 90}          (deg for revolute, m for prismatic)
//    "motor": {"velocity": 1.0, "max_force": 50}    (revolute: rad/s and N m)
//    "spring": {"stiffness": 1e4, "damping": 100},  (distance joints: a spring between min and max)
//    "min_distance": 0, "max_distance": 1.5,        (distance joints)
//    "break_force": 5e4, "break_torque": 1e4,       (N, N m; the joint snaps beyond these)
//    "wind_rotor": {"radius_m": 40, "cp": 0.45, "tip_speed_ratio": 7}}   (revolute: the wind turns it, below)
// Types: fixed, revolute, prismatic, spherical, distance. Verbs: release (detach now), motor {velocity, max_force}.
// A wind rotor (turbine, windmill, pinwheel) feels the wind along its axis with a torque that falls linearly from
// stall to its runaway speed; it delivers its rated power cp * rho/2 * pi R^2 v^3 at the given tip speed ratio.
#pragma once

#include <optional>

#include "world/World.h"

namespace p2s {

class Joint : public Entity {
public:
    Joint(const Spec& s, World& w);

    void link(World& w) override;
    void beforeStep(World& w, double dt) override;
    void afterStep(World& w, double dt) override;
    double speed() const;  // revolute: rad/s of b relative to a about the axis; 0 otherwise or once broken
    bool isRevolute() const { return type_ == "revolute"; }
    void record(World& w, Recorder& rec) override;
    Json describe() const override;
    std::vector<std::string> verbs() const override { return {"release", "motor"}; }
    void act(World& w, const Action& a) override;

private:
    struct Range {
        double lower, upper;
    };
    struct Motor {
        double velocity, maxForce;
    };
    struct Spring {
        double stiffness, damping;
    };
    struct WindRotor {
        double radius, cp, tsr;
    };

    physx::PxRigidActor* actorOf(World& w, const std::string& id);  // PhysX joints take actor pointers; null = world
    void startMotor(World& w, const Motor& m);

    std::string type_, a_, b_, motorId_;
    Vec3 anchor_, axis_{1, 0, 0};
    std::optional<Range> limits_;
    std::optional<Motor> motor_;
    std::optional<Spring> spring_;
    std::optional<WindRotor> windRotor_;
    std::string windId_;
    std::optional<double> minDistance_, maxDistance_;
    double breakForce_ = 0, breakTorque_ = 0;
    PxPtr<physx::PxJoint> joint_;
    bool broken_ = false;
};

}  // namespace p2s
