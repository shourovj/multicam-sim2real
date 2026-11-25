// A tracked vehicle (tank, bulldozer): a rigid hull on low-friction track boxes, driven by a traction
// controller that does what the tracks would: drive along the heading, resist sideslip and yaw, all capped by
// track-ground friction and only while the tracks touch the ground (from the tank-truck crash).
//
//   all rigid-body keys (shapes with "material": "track" for the track boxes, pos, mass_kg, ...) plus
//   "traction": {"speed_mps": 5, "mu": 0.7, "track_half_length_m": 1.8, "response_s": 0.25,
//                "path": [[x, y], ...], "lat_accel_max": 2.0}     optional route (else it holds its heading)
// Verbs: rigid-body verbs plus brake (target speed 0) and set_speed {mps}.
#pragma once

#include <optional>

#include "bodies/RigidBody.h"
#include "control/Driver.h"

namespace p2s {

class TrackedVehicle : public RigidBody {
public:
    TrackedVehicle(const Spec& s, World& w);

    void link(World& w) override;
    void beforeStep(World& w, double dt) override;
    void record(World& w, Recorder& rec) override;
    Json describe() const override;
    std::vector<std::string> verbs() const override;
    void act(World& w, const Action& a) override;

private:
    double target_ = 0, mu_ = 0.7, halfLength_ = 1.8, response_ = 0.25, groundZ_ = 0;
    double lastForce_ = 0;
    bool onGround_ = true;
    std::optional<Polyline> path_;  // optional route: the tank steers along it and slows for bends
    std::vector<double> limit_;
};

}  // namespace p2s
