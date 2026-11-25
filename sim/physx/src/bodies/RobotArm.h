// A robot arm (PhysX articulation from sim/physx/robots/<robot>.json): joint position drives follow inverse
// kinematics of a keyframed target, links are gravity-compensated as on a real arm, and the held object hangs on
// a fixed joint at the tool centre point, so it is carried (and its contents slosh) physically.
//
//   {"id": "robot", "robot": "franka_panda", "base": {"pos": [x, y, z], "yaw_deg": 0},
//    "hold": {"object": "pitcher", "offset": {"pos": [0.08, 0, 0], "pitch_deg": -90}, "gripper_m": 0.035},
//    "point_local": [0.047, 0, 0.16],      the keyed point, in the held object's frame (or the TCP's if no hold)
//    "keys": [{"t": 0, "point": [x, y, z], "yaw_deg": -90, "tilt_deg": 0}, ...],   object yaw, then tilt about its Y
//    "drive": {"stiffness": 2000, "damping": 200, "max_torque_nm": 87}}
// Streams: robot/<id>/links (pose of every link), robot/<id>/joints (joint values, IK error m, deg).
#pragma once

#include <optional>

#include "control/Kinematics.h"
#include "world/World.h"

namespace p2s {

class RobotArm : public Entity {
public:
    RobotArm(const Spec& s, World& w);
    ~RobotArm() override;

    void link(World& w) override;
    void beforeStep(World& w, double dt) override;
    void record(World& w, Recorder& rec) override;
    Json describe() const override;
    Energy energy(const World& w) const override;
    bool isRigid() const override { return true; }
    physx::PxRigidActor& actor() override;  // the hand link
    Pose pose() const override { return base_; }

private:
    struct Key {
        double t;
        Vec3 point;
        double yaw, tilt;
    };
    Pose targetTcp(double t) const;
    void readJoints();

    std::string robotName_, holdId_;
    Chain chain_;
    Pose base_, holdOffset_;
    Vec3 pointLocal_;
    double gripper_ = 0.04, stiffness_ = 2000, damping_ = 200, maxTorque_ = 87;
    std::vector<Key> keys_;

    PxPtr<physx::PxArticulationReducedCoordinate> articulation_;
    PxPtr<physx::PxArticulationCache> cache_;
    PxPtr<physx::PxJoint> grip_;
    std::vector<std::reference_wrapper<physx::PxArticulationLink>> links_;
    std::vector<int> dofStart_;          // cache index of each link's joint value (-1 if none)
    std::vector<double> armQ_, q_, qd_;  // IK solution (arm joints); measured values per link
    double ikErrM_ = 0, ikErrDeg_ = 0;
};

}  // namespace p2s
