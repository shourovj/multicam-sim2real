// Particle matter on PhysX's GPU position-based dynamics: liquids (water, oil, honey), granular media (sand,
// gravel) and smoke. Particles collide with every rigid shape and deformable in the scene.
//
//   {"id": "oil", "kind": "fluid" | "granular" | "smoke", "preset": "water" | "oil" | "honey" | "sand" | "smoke",
//    "spacing_m": 0.012, "max_particles": 30000,
//    material overrides: "density", "viscosity", "cohesion", "adhesion", "surface_tension", "friction", "damping",
//    smoke only: "buoyancy" (upward acceleration as a fraction of g), "lifetime_s", "drag" (toward the wind),
//                "diffusion_mps" (random mixing velocity that widens the plume, 0.4),
//    "blocks": [{"min": [x, y, z], "max": [x, y, z], "velocity": [...]}],      volumes filled at t = 0 (world)
//    "emitters": [{"host": "truck", "pos": [x, y, z], "direction": [0, 0, -1],  host-local (or world)
//                  "radius_m": 0.05, "speed_mps": 1.6, "active": false,          start with a trigger: do "start"
//                  "phases": [{"after_s": 0.5, "radius_m": 0.022, "speed_mps": 1.0}]}]}
// Verbs: start, stop (all emitters). Stream particles/<id>/pos: variable rows per frame (alive particles).
#pragma once

#include <optional>

#include "physx/Pinned.h"
#include "world/World.h"

namespace p2s {

class ParticleSystem : public Entity {
public:
    ParticleSystem(const Spec& s, World& w);

    void link(World& w) override;
    void beforeStep(World& w, double dt) override;
    void afterStep(World& w, double dt) override;
    void record(World& w, Recorder& rec) override;
    Json describe() const override;
    std::vector<std::string> verbs() const override { return {"start", "stop"}; }
    void act(World& w, const Action& a) override;
    Energy energy(const World& w) const override;
    std::span<const Vec3> contactPoints() const override { return alivePoints_; }
    double contactRadius() const override { return restOffset_; }

    // A blast wave: particles within `radius` gain `speed` away from `centre`, fading linearly to the edge.
    // Returns the kinetic energy added (0 for smoke, whose energy is the air's).
    double kick(World& w, const Vec3& centre, double radius, double speed);

private:
    enum class Kind { Fluid, Granular, Smoke };
    struct Phase {
        double afterS, radius, speed;
    };
    struct Emitter {
        std::string host;
        Vec3 pos, dir;
        double radius, speed;
        bool active;
        std::vector<Phase> phases;
        double startedAt = -1, lastLayer = -1e9;
        std::optional<std::reference_wrapper<Entity>> hostRef;
    };
    struct Block {
        Aabb box;
        Vec3 velocity;
    };

    void readBack(World& w);
    void upload(World& w);
    bool emit(World& w, Emitter& e);
    int freeSlot();

    Kind kind_ = Kind::Fluid;
    std::string preset_;
    double spacing_ = 0.012, density_ = 1000, viscosity_ = 0.05, cohesion_ = 0.05, adhesion_ = 0.0, diffusion_ = 0;
    double surfaceTension_ = 0.02, friction_ = 0.1, damping_ = 0.0, buoyancy_ = 0, lifetime_ = 1e9, drag_ = 0;
    int max_ = 30000, highWater_ = 0, alive_ = 0;
    std::vector<Block> blocks_;
    std::vector<Emitter> emitters_;

    PxPtr<physx::PxPBDParticleSystem> system_;
    PxPtr<physx::PxPBDMaterial> material_;
    PxPtr<physx::PxParticleBuffer> buffer_;
    Pinned<physx::PxVec4> pos_, vel_;
    std::vector<double> born_;  // birth time per slot; < -1e8 = free
    std::vector<int> free_;     // free slots, lowest index last
    std::vector<Vec3> alivePoints_;
    float invMass_ = 1, restOffset_ = 0;  // restOffset_: particle centre to solid surface at rest
    uint32_t phase_ = 0;
    double emitted_ = 0, kinetic_ = 0, potential_ = 0;
    bool dirty_ = false;
};

}  // namespace p2s
