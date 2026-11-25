// Cloth (PhysX FEM deformable surface on the GPU): flags, banners, tarps, sheets. Wind acts per triangle as
// quadratic normal drag plus tangential friction (ported from the tank-truck flag).
//
//   {"id": "flag", "size_m": [2.4, 1.6], "grid": [41, 28],       rectangle in its local x-y plane
//    "pos": [...], "yaw_deg": ..., "roll_deg": 90,               pose of the rectangle's (0, 0) corner
//    "host": "pole",                                              optional: pins move with this body
//    "pin": [{"min": [0, 0], "max": [0.001, 1.6]}],               local (x, y) boxes of pinned vertices
//    "areal_density": 0.25 (kg/m^2), "thickness_m": 0.002, "youngs_pa": 5e6, "poisson": 0.3,
//    "bending": 2e-6, "friction": 0.5, "damping": 0.02, "drag": {"normal": 1.2, "tangent": 0.15},
//    "solver_iterations": 30}
// Verbs: release (the pins let go: a flag tears off its pole, a sheet drops).
// Stiff thin sheets (a playing card, a sheet of card, a plastic panel) are cloths with a high "bending".
// Streams: cloth/<id>/verts (world, every frame); static: triangles, uv.
#pragma once

#include "physx/Pinned.h"
#include "world/World.h"

namespace p2s {

class Cloth : public Entity {
public:
    Cloth(const Spec& s, World& w);

    void link(World& w) override;
    void beforeStep(World& w, double dt) override;
    void afterStep(World& w, double dt) override;
    void record(World& w, Recorder& rec) override;
    Json describe() const override;
    Energy energy(const World& w) const override;
    std::vector<std::string> verbs() const override { return {"release"}; }
    void act(World& w, const Action& a) override;
    std::span<const Vec3> contactPoints() const override { return current_; }
    double contactRadius() const override { return thickness_; }
    const std::string& contactAlias() const override { return hostId_; }

private:
    void readBack(World& w);
    void applyWind(World& w, double dt);

    double width_ = 1, height_ = 1, areal_ = 0.25, thickness_ = 0.002, youngs_ = 5e6, poisson_ = 0.3;
    double bending_ = 2e-6, friction_ = 0.5, damping_ = 0.02, cn_ = 1.2, ct_ = 0.15;
    int nu_ = 21, nv_ = 14, iterations_ = 30;
    Pose pose_;
    std::string hostId_;
    std::vector<Aabb> pins_;

    PxPtr<physx::PxTriangleMesh> mesh_;  // must outlive the surface
    PxPtr<physx::PxDeformableSurface> surface_;
    PxPtr<physx::PxDeformableSurfaceMaterial> material_;
    PxPtr<physx::PxShape> pxShape_;
    PxPtr<physx::PxDeformableAttachment> attachment_;
    Pinned<physx::PxVec4> pos_, vel_, rest_;
    std::vector<std::array<int32_t, 3>> triangles_;
    std::vector<float> uv_;
    std::vector<Vec3> current_;
    int nVerts_ = 0, nPinned_ = 0;
    double kinetic_ = 0, potential_ = 0;
};

}  // namespace p2s
