// Joints between two rigid entities (or one entity and the world).
//
//   {"id": "hinge", "type": "revolute", "a": "frame" | "world", "b": "door",
//    "anchor": [x, y, z] (world, at t = 0), "axis": [0, 0, 1] (world; revolute/prismatic),
//    "limits": {"lower": -90, "upper": 90}          (deg for revolute, m for prismatic)
//    "motor": {"velocity": 1.0, "max_force": 50}    (revolute: rad/s and N m)
//    "spring": {"stiffness": 1e4, "damping": 100},  (distance joints: a spring between min and max)
//    "min_distance": 0, "max_distance": 1.5,        (distance joints)
//    "break_force": 5e4, "break_torque": 1e4}       (N, N m; the joint snaps beyond these)
// Types: fixed, revolute, prismatic, spherical, distance. Verbs: release (detach now), motor {velocity, max_force}.
#pragma once

#include <optional>

#include "world/World.h"

namespace p2s {

class Joint : public Entity {
public:
    Joint(const Spec& s, World& w);

    void link(World& w) override;
    void afterStep(World& w, double dt) override;
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

    physx::PxRigidActor* actorOf(World& w, const std::string& id);  // PhysX joints take actor pointers; null = world
    void startMotor(World& w, const Motor& m);

    std::string type_, a_, b_, motorId_;
    Vec3 anchor_, axis_{1, 0, 0};
    std::optional<Range> limits_;
    std::optional<Motor> motor_;
    std::optional<Spring> spring_;
    std::optional<double> minDistance_, maxDistance_;
    double breakForce_ = 0, breakTorque_ = 0;
    PxPtr<physx::PxJoint> joint_;
    bool broken_ = false;
};

}  // namespace p2s
