// A pedestrian (from sim/physics/pedestrian.py): a kinematic capsule that walks a path at a steady pace or
// stands. Others collide with it; it is not pushed back. The renderer poses a human from walk_phase.
//   walking:  {"id", "path": [[x, y], ...], "speed_mps": 1.4, "start_s": 0, "start_offset_m": 0}
//   standing: {"id", "idle_at": [x, y], "facing_deg": 0, "pose": "stand" | "phone", "sway_m": 0.02}
//   common:   "height_m": 1.75, "radius_m": 0.25, "ground_z": 0
// Stream walker/<id>: walk_phase (0..1 per stride), walking (0/1), speed, pose_id (0 walk, 1 stand, 2 phone).
#pragma once

#include "bodies/RigidBody.h"
#include "control/Driver.h"

namespace p2s {

class Walker : public Entity {
public:
    Walker(const Spec& s, World& w);

    void beforeStep(World& w, double dt) override;
    void afterStep(World& w, double dt) override;
    void record(World& w, Recorder& rec) override;
    Json describe() const override;
    bool isRigid() const override { return true; }
    physx::PxRigidActor& actor() override { return *actor_; }
    const std::vector<Shape>& shapes() const override { return shapes_; }
    Pose pose() const override { return poseAt(t_); }
    Vec3 velocity() const override;
    bool walking(double t) const;

private:
    Pose poseAt(double t) const;
    double distance(double t) const;

    Polyline path_;  // walking
    Vec3 idleAt_;    // standing
    double speed_ = 1.4, startS_ = 0, offset_ = 0, height_ = 1.75, radius_ = 0.25, groundZ_ = 0, sway_ = 0.02;
    double yaw0_ = 0, t_ = 0;
    std::string mode_ = "walk";
    std::vector<Shape> shapes_;
    PxPtr<physx::PxRigidDynamic> actor_;
};

}  // namespace p2s
