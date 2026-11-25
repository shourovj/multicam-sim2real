// A magnet (crane electromagnet, fridge magnet, a lodestone) pulling or pushing listed iron bodies.
//
//   {"id": "crane_magnet", "host": "magnet_plate" | "world", "pos": [x, y, z],   host-local (world if no host)
//    "targets": ["scrap1", "scrap2"],              bodies the field acts on (only iron and steel are magnetic)
//    "force_n_at_1m": 2000,                        pull on each target 1 m from its nearest point
//    "max_force_n": 10 x force_n_at_1m,            cap close up (a pole face saturates)
//    "falloff": 2,                                 force ~ 1/d^falloff (2: pole, 4: dipole far field)
//    "repel": false, "active": true}
// The host feels the opposite force (Newton's third law). Verbs: on, off, repel, attract.
#pragma once

#include <optional>

#include "world/World.h"

namespace p2s {

class Magnet : public Entity {
public:
    Magnet(const Spec& s, World& w);

    void link(World& w) override;
    void beforeStep(World& w, double dt) override;
    void record(World& w, Recorder& rec) override;
    Json describe() const override;
    std::vector<std::string> verbs() const override { return {"on", "off", "repel", "attract"}; }
    void act(World& w, const Action& a) override;

private:
    Vec3 worldPos() const;

    std::string host_;
    std::vector<std::string> targetIds_;
    std::optional<std::reference_wrapper<Entity>> hostRef_;
    std::vector<std::reference_wrapper<Entity>> targets_;
    Vec3 pos_;
    double k_ = 0, max_ = 0, falloff_ = 2, peak_ = 0;
    bool repel_ = false, active_ = true;
};

}  // namespace p2s
