// A wheeled vehicle on the PhysX Vehicle SDK: rigid chassis, raycast suspension (spring + damper) per wheel,
// slip-based tyre forces, direct drive to the driven wheels, brakes, Ackermann steering. A Driver (lane path,
// IDM, signals, crosswalks) or triggers work the controls; the tyres produce every force.
//
//   {"id": "car", "pos": [x, y, 0], "yaw_deg": 90, "speed_mps": 10,
//    "mass_kg": 1500, "shapes": [...chassis collision shapes, body frame: +X forward, +Z up, origin on the ground],
//    "wheels": {"radius_m": 0.33, "half_width_m": 0.11, "mass_kg": 20,
//               "axles": [{"x": 1.35, "track_m": 1.6, "steer": true, "drive": false},
//                         {"x": -1.35, "track_m": 1.6, "steer": false, "drive": true}]},   front to back
//    "suspension": {"travel_m": 0.2, "frequency_hz": 1.6, "damping_ratio": 0.3},
//    "tyre_friction": 0.9,
//    "limits": {"max_wheel_torque_nm": 800, "max_brake_torque_nm": 2500, "max_steer_deg": 30},
//    "com": [x, y, z], "driver": {...see Driver...}}
// Verbs: brake {level 0..1}, release (driver lets go of everything), set_speed {mps}.
#pragma once

#include <optional>

#include "bodies/RigidBody.h"
#include "control/Driver.h"
#include "snippetvehiclecommon/directdrivetrain/DirectDrivetrain.h"
#include "world/World.h"

namespace p2s {

class Vehicle : public Entity {
public:
    Vehicle(const Spec& s, World& w);
    ~Vehicle() override;

    void link(World& w) override;
    void beforeStep(World& w, double dt) override;
    void afterStep(World& w, double dt) override;
    void record(World& w, Recorder& rec) override;
    Json describe() const override;
    std::vector<std::string> verbs() const override { return {"brake", "release", "set_speed"}; }
    void act(World& w, const Action& a) override;
    Energy energy(const World& w) const override;

    bool isRigid() const override { return true; }
    physx::PxRigidActor& actor() override { return chassis(); }
    const std::vector<Shape>& shapes() const override { return shapes_; }
    Pose pose() const override;
    Vec3 velocity() const override;

private:
    physx::PxRigidDynamic& chassis();
    const physx::PxRigidDynamic& chassis() const;
    Pose wheelPose(int i) const;
    void logActuators(World& w, double dt);

    snippetvehicle::DirectDriveVehicle vehicle_;
    physx::PxVehiclePhysXSimulationContext ctx_;
    physx::PxVehiclePhysXMaterialFriction friction_;
    std::vector<Shape> shapes_;
    std::optional<Driver> driver_;
    VehicleLimits limits_{};
    int nWheels_ = 0;
    double wheelRadius_ = 0.33, wheelHalfWidth_ = 0.11;
    MotionPeaks peaks_;
    std::string driveId_, brakesId_;  // actuator ids
    double manualBrake_ = -1;
    bool released_ = false, crashRelease_ = true, crashed_ = false;
    DriveCommand lastCmd_;
};

}  // namespace p2s
