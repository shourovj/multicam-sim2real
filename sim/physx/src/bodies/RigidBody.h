// Rigid bodies: dynamic (forces and contacts move it), kinematic (follows keyframes, pushes others) or static.
//
//   {"id": "crate", "type": "dynamic", "shapes": [{"type": "box", "size": [1, 1, 1], "material": "wood"}],
//    "pos": [0, 0, 2], "yaw_deg": 30, "mass_kg": 20 | "density": 600,
//    "velocity": [0, 0, 0], "angular_velocity": [0, 0, 0] (rad/s, world),
//    "drag": {"linear_per_s": 0.3} | {"cd": 1.05, "area_m2": 1.0},  (quadratic drag uses the wind field)
//    "damping": {"linear": 0, "angular": 0.05}, "ccd": false, "assembly": "truck",
//    "keys": [{"t": 0, "pos": [...], "yaw_deg": 0}, ...]           (kinematic: smooth path through keys)
//    "thrusters": [...], "hull_cd": 0.8                              (bodies/Propulsion.h)
//   }
// Verbs: impulse, push, set_velocity, release, freeze, ignite/cutoff {thruster}.
#pragma once

#include <optional>
#include <set>

#include "bodies/Propulsion.h"
#include "world/World.h"

namespace p2s {

struct PoseKey {
    double t;
    Pose pose;
};

// Smoothstep-eased interpolation between pose keys (position lerp, rotation slerp); holds the ends.
Pose interpolateKeys(const std::vector<PoseKey>& keys, double t);
std::vector<PoseKey> parseKeys(const Spec& s, const std::string& key = "keys");

// Kinetic (translation + rotation about the centre of mass) and gravitational energy of a PhysX body.
Energy rigidEnergy(const physx::PxRigidDynamic& d, const Vec3& gravity);
// Kinetic energy an impulse J at world point p adds to a rigid body (translation and rotation).
double impulseWork(const physx::PxRigidDynamic& d, const Vec3& J, const Vec3& p);

// Peak speed and peak acceleration |dv/dt| (0 at rest, 1 g in free fall, the deceleration in a crash), per step.
struct MotionPeaks {
    Vec3 lastVel;
    double accel = 0, speed = 0;
    void update(const Vec3& v, double dt, const Vec3& /*gravity*/) {
        accel = std::max(accel, ((v - lastVel) / dt).norm());
        speed = std::max(speed, v.norm());
        lastVel = v;
    }
};

class RigidBody : public Entity {
public:
    RigidBody(const Spec& s, World& w, std::string kind = "rigid_body");
    // Static, from code (the ground). `category` is the collision category bit (2 = ground).
    RigidBody(std::string id, std::vector<Shape> shapes, const Pose& pose, World& w, uint32_t category = 1);

    void link(World& w) override;
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
    Vec3 velocity() const override;
    physx::PxRigidDynamic& dynamic();  // throws for static bodies

private:
    void applyDrag(World& w);
    void scriptedWork(World& w, double joules);

    std::string type_;
    std::vector<Shape> shapes_;
    PxPtr<physx::PxRigidActor> actor_;
    std::vector<PoseKey> keys_;
    double linearDrag_ = 0, cd_ = 0, area_ = 0;
    MotionPeaks peaks_;
    struct Push {
        Vec3 force;
        double until;
    };
    std::optional<Push> push_;
    std::vector<Thruster> thrusters_;
    std::optional<Buoyancy> buoyancy_;  // only when the world has water
    double submerged_ = 0;  // fraction under water, last step
    bool ccd_ = false;  // PhysX allows CCD on dynamic bodies only: applied on release
};

// The ground plane (z = height), created from world.ground.
std::unique_ptr<Entity> makeGround(const Spec& s, World& w);

}  // namespace p2s
