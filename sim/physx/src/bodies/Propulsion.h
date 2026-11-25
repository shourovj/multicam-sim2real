// Forces a rigid body can carry besides contacts: buoyancy and drag in water, and thrusters (rocket motors, jets,
// ship propellers, bow thrusters).
//
//   "thrusters": [{"id": "main", "pos": [x, y, z], "direction": [1, 0, 0],   body frame; force pushes along direction
//                  "force_n": 5000,                                          constant thrust while burning, or
//                  "joint": "prop_shaft", "k": 0.8,                          thrust k w|w| from a propeller joint's spin
//                  "burn_s": [t0, t1], "active": true,                       on between t0 and t1 / from the start
//                  "medium": "any" | "water" | "air"}]                       propellers push only where they are wet
//   Water (world.water): each shape is split into small cells; a wet cell gets buoyancy rho g V at its centre (so a
//   hull rights itself) and quadratic drag against the current. "hull_cd" (0.8) sets that drag.
#pragma once

#include <optional>

#include "core/Water.h"
#include "physx/Px.h"
#include "world/World.h"

namespace p2s {

class Joint;

class Buoyancy {
public:
    Buoyancy(const std::vector<Shape>& shapes, double dragCd);
    struct Result {
        double submerged = 0;  // fraction of the body's volume under water
        double power = 0;      // W the water did on the body (lift while rising, drag), for the energy budget
    };
    // Applies the forces for this step.
    Result apply(physx::PxRigidDynamic& body, const Water& water, const Vec3& gravity) const;

private:
    struct Cell {
        Vec3 local;  // body frame
        double volume, area;
    };
    std::vector<Cell> cells_;
    double cd_;
};

class Thruster {
public:
    explicit Thruster(const Spec& s);
    void link(World& w, const std::string& owner, double ownerMass);
    // Applies thrust for this step and logs it; `w.water()` decides whether a water-only thruster bites.
    void apply(physx::PxRigidDynamic& body, World& w, double dt);
    const std::string& id() const { return id_; }
    void setActive(bool on) { active_ = on; }
    Json toJson() const;

private:
    std::string id_, jointId_, medium_, actuatorId_;
    Vec3 pos_, dir_;
    double force_ = 0, k_ = 0, last_ = 0;
    std::optional<std::array<double, 2>> burn_;
    bool active_ = true;
    std::optional<std::reference_wrapper<Joint>> joint_;
};

}  // namespace p2s
