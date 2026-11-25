// A driver that follows a lane path: pure-pursuit steering, Intelligent Driver Model speed (bends, leader,
// red lights, pedestrians on crosswalks), and throttle/brake commands for a wheeled vehicle. Ports
// sim/physics/ground_vehicle.py and lane_vehicle.py onto a real vehicle.
//
//   "driver": {"path": [[x, y], ...], "cruise_mps": 10,
//              "accel_max": 2.0, "brake_comfort": 2.0, "brake_max": 6.0, "brake_yellow": 3.0,
//              "lat_accel_max": 2.5, "headway_s": 1.2, "gap_min_m": 2.0,
//              "leader": "car_a", "signal": "tl_NS", "stop_s": 31.0,
//              "yield_to": ["ped_a"], "crosswalks": [[s0, s1], ...], "lane_width_m": 3.5}
// Arc lengths (stop_s, crosswalks) are metres along `path` from its first point.
#pragma once

#include <optional>

#include "world/World.h"

namespace p2s {

class TrafficSignal;
class Walker;

struct DriveCommand {
    double throttle = 0, brake = 0, steer = 0;  // throttle, brake in [0, 1]; steer in [-1, 1] (+ = left)
};

struct VehicleLimits {
    double massKg, maxDriveForceN, maxBrakeForceN, wheelbaseM, maxSteerRad, lengthM;
};

class Polyline {
public:
    Polyline() = default;
    explicit Polyline(std::vector<std::array<double, 2>> pts);
    // From a scene list [[x, y], ...] of at least two points.
    static Polyline parse(const Spec& list);
    double length() const { return s_.back(); }
    Vec3 at(double s) const;          // point (z = 0)
    Vec3 tangent(double s) const;
    double project(const Vec3& p) const;  // arc length of the closest point
    double lateral(const Vec3& p) const;  // signed distance, + to the left of the path
    // Highest speed allowed along the path: the bend limit, lowered ahead of bends for comfortable braking.
    std::vector<double> speedLimit(double cruise, double latAccel, double brake) const;
    const std::vector<double>& arcs() const { return s_; }

private:
    std::vector<std::array<double, 2>> p_;
    std::vector<double> s_;
};

class Driver {
public:
    explicit Driver(const Spec& s);
    void link(World& w, const std::string& owner);

    // Commands for the next step of length dt; `speed` is the forward speed of the vehicle at `pose` (its centre).
    DriveCommand update(World& w, const std::string& owner, const Pose& pose, double speed, const VehicleLimits& lim,
                        double dt);
    void setCruise(double v) { cruise_ = v; limit_ = path_.speedLimit(cruise_, latAccel_, brakeComfort_); }
    double arc() const { return s_; }
    Json toJson() const;

private:
    double idmAccel(World& w, double front, double speed, const VehicleLimits& lim);
    bool walkerInBand(const Walker& p, double s0, double s1) const;

    Polyline path_;
    std::vector<double> limit_;
    double cruise_ = 10, accelMax_ = 2, brakeComfort_ = 2, brakeMax_ = 6, brakeYellow_ = 3, latAccel_ = 2.5;
    double headway_ = 1.2, gapMin_ = 2, laneWidth_ = 3.5;
    std::optional<double> stopS_;
    std::vector<std::array<double, 2>> crosswalks_;
    std::string leaderId_, signalId_;
    std::vector<std::string> yieldIds_;
    std::optional<std::reference_wrapper<Entity>> leader_;
    std::optional<std::reference_wrapper<TrafficSignal>> signal_;
    std::vector<std::reference_wrapper<Walker>> yield_;
    std::optional<bool> yellowGo_;
    double s_ = 0, targetSpeed_ = 0;
    bool stopped_ = false, passedStop_ = false, initialised_ = false;
};

}  // namespace p2s
