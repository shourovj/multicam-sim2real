// A blast: at one moment it throws nearby bodies and particles outward. Only its effect on the surroundings is
// simulated; the fireball, flash and smoke are left to the video model.
//
//   {"id": "blast", "pos": [x, y, z], "at_s": 1.2,          (or no at_s and a trigger doing "detonate")
//    "radius_m": 8,                                          reach of the blast wave
//    "impulse_ns": 4000,                                     impulse on a 1 m^2 face 1 m away (falls with 1/d^2)
//    "particle_speed_mps": 12}                               speed given to particles at the centre (falls to 0 at
//                                                            the edge): sand, water and smoke are thrown too
// Each dynamic rigid body within radius_m gets J = impulse_ns * A / max(d, 1)^2 away from the centre, applied at its
// nearest point (so it also spins), where A is its area facing the blast and d the distance to that point.
// Events: <id>:detonated.
#pragma once

#include <optional>

#include "world/World.h"

namespace p2s {

class Explosion : public Entity {
public:
    Explosion(const Spec& s, World& w);

    void link(World& w) override;
    void beforeStep(World& w, double dt) override;
    void record(World&, Recorder&) override {}
    Json describe() const override;
    std::vector<std::string> verbs() const override { return {"detonate"}; }
    void act(World& w, const Action& a) override;

private:
    void detonate(World& w, double dt);

    Vec3 pos_;
    std::optional<double> at_;
    double radius_ = 8, impulse_ = 4000, particleSpeed_ = 12;
    std::optional<double> detonatedAt_;
    double peakImpulse_ = 0;
};

}  // namespace p2s
