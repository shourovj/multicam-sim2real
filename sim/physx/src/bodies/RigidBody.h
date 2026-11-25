// Rigid bodies: dynamic (forces and contacts move it), kinematic (follows keyframes, pushes others) or static.
//
//   {"id": "crate", "type": "dynamic", "shapes": [{"type": "box", "size": [1, 1, 1], "material": "wood"}],
//    "pos": [0, 0, 2], "yaw_deg": 30, "mass_kg": 20 | "density": 600,
//    "velocity": [0, 0, 0], "angular_velocity": [0, 0, 0] (rad/s, world),
//    "drag": {"linear_per_s": 0.3} | {"cd": 1.05, "area_m2": 1.0},  (quadratic drag uses the wind field)
//    "damping": {"linear": 0, "angular": 0.05}, "ccd": false, "assembly": "truck",
//    "keys": [{"t": 0, "pos": [...], "yaw_deg": 0}, ...]           (kinematic: smooth path through keys)
//   }
#pragma once

#include <optional>

#include "world/World.h"

namespace p2s {

struct PoseKey {
    double t;
    Pose pose;
};

// Smoothstep-eased interpolation between pose keys (position lerp, rotation slerp); holds the ends.
Pose interpolateKeys(const std::vector<PoseKey>& keys, double t);
std::vector<PoseKey> parseKeys(const Spec& s, const std::string& key = "keys");

class RigidBody : public Entity {
public:
    RigidBody(const Spec& s, World& w);
    RigidBody(std::string id, std::vector<Shape> shapes, const Pose& pose, World& w);  // static, from code (ground)

    void beforeStep(World& w, double dt) override;
    void afterStep(World& w, double dt) override;
    void record(World& w, Recorder& rec) override;
    Json describe() const override;
    std::vector<std::string> verbs() const override;
    void act(World& w, const Action& a) override;
    Energy energy(const World& w) const override;

    bool isRigid() const override { return true; }
    physx::PxRigidActor& actor() override { return *actor_; }
    const std::vector<Shape>& shapes() const override { return shapes_; }
    Pose pose() const override { return fromPx(actor_->getGlobalPose()); }

    double mass() const;
    Vec3 velocity() const;
    physx::PxRigidDynamic& dynamic();  // throws for static bodies

private:
    void applyDrag(World& w);

    std::string type_;
    std::vector<Shape> shapes_;
    PxPtr<physx::PxRigidActor> actor_;
    std::vector<PoseKey> keys_;
    double linearDrag_ = 0, cd_ = 0, area_ = 0;
    Vec3 lastVel_;
    double peakAccel_ = 0, peakSpeed_ = 0;
    struct Push {
        Vec3 force;
        double until;
    };
    std::optional<Push> push_;
};

// The ground plane (z = height), created from world.ground.
std::unique_ptr<Entity> makeGround(const Spec& s, World& w);

}  // namespace p2s
