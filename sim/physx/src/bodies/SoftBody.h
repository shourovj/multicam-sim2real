// A deformable solid (PhysX FEM on the GPU): elastic, or plastic so that it stays crushed. Ported from the
// tank-truck crash (truck front end, armour panel), where it absorbs impact energy.
//
//   {"id": "truck_front", "host": "truck",              optional; geometry and attach boxes are then host-local
//    "shape": {...one closed shape: box, prism, convex, mesh, sphere, capsule, cylinder...},
//    "pos": [...], "yaw_deg": ...,                       world pose when there is no host
//    "mass_kg": 1500 | "density": 800,
//    "youngs_pa": 2e7, "poisson": 0.3, "friction": 0.5, "damping": 0.005,
//    "plastic": {"yield_strain": 0.001, "low_limit": 0.2, "high_limit": 5.0},   omit for elastic
//    "attach": [{"min": [x, y, z], "max": [x, y, z]}, ...],   vertices in these boxes are bolted to the host
//                                                              (or pinned to the world without a host)
//    "resolution": 20 (voxels across the longest side), "max_edge_m": 0.12, "solver_iterations": 30,
//    "velocity": [...], "max_depenetration_mps": 2.0}
// Streams: softbody/<id>/verts (world positions of the collision mesh, every frame); static: tets, triangles.
#pragma once

#include <optional>

#include "physx/Pinned.h"
#include "world/World.h"

namespace p2s {

class SoftBody : public Entity {
public:
    SoftBody(const Spec& s, World& w);

    void link(World& w) override;
    void beforeStep(World& w, double dt) override;
    void afterStep(World& w, double dt) override;
    void record(World& w, Recorder& rec) override;
    Json describe() const override;
    Energy energy(const World& w) const override;
    std::span<const Vec3> contactPoints() const override { return surface_; }
    std::string contactAlias() const override { return hostId_; }

private:
    void build(World& w);
    void readBack(World& w, bool velocities);

    Json shapeSpec_;
    Shape shape_;
    Pose pose_;
    std::string hostId_;
    std::optional<double> mass_, density_;
    double youngs_ = 1e6, poisson_ = 0.3, friction_ = 0.5, damping_ = 0.005;
    std::optional<std::array<double, 3>> plastic_;  // yield strain, low limit, high limit
    std::vector<Aabb> attach_;
    int resolution_ = 20, iterations_ = 30;
    double maxEdge_ = 0.12, maxDepen_ = 2.0;
    Vec3 velocity0_;

    PxPtr<physx::PxDeformableVolumeMesh> mesh_;  // PhysX keeps using the tet meshes: must outlive body_
    PxPtr<physx::PxDeformableVolume> body_;
    PxPtr<physx::PxDeformableVolumeMaterial> material_;
    PxPtr<physx::PxShape> pxShape_;
    PxPtr<physx::PxDeformableAttachment> attachment_;
    Pinned<physx::PxVec4> simPos_, simVel_, collPos_;  // host mirrors of the GPU state
    std::vector<int32_t> tets_, triangles_;   // collision mesh
    std::vector<int> surfaceIndex_;           // collision vertices on the surface
    std::vector<Vec3> rest_, current_, surface_;
    double kinetic_ = 0, potential_ = 0, peakStrain_ = 0, volume_ = 0;
    int nSim_ = 0, nColl_ = 0, nAttached_ = 0;
};

}  // namespace p2s
