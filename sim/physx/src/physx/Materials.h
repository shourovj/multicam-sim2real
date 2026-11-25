// Named surface materials. Presets cover the common cases so a scene rarely needs raw coefficients:
//   default, steel, aluminium, rubber, tyre, wood, concrete, asphalt, ice, glass, plastic, ceramic, flesh, track, cloth
#pragma once

#include <map>
#include <string>

#include "core/Spec.h"
#include "physx/Px.h"

namespace p2s {

struct MaterialProps {
    double staticFriction = 0.5, dynamicFriction = 0.4, restitution = 0.1;
    double density = 1000;  // kg/m^3, used when a body gives no mass
    std::string frictionCombine = "average", restitutionCombine = "average";
};

class MaterialLibrary {
public:
    explicit MaterialLibrary(physx::PxPhysics& physics);

    // scene "materials": [{id, preset?, static_friction?, dynamic_friction?, restitution?, density?, ...}]
    void add(const Spec& s);
    physx::PxMaterial& get(const std::string& id, const std::string& usedBy) const;
    const MaterialProps& props(const std::string& id) const;
    Json toJson() const;

    static const std::map<std::string, MaterialProps>& presets();

private:
    void create(const std::string& id, const MaterialProps& p);

    physx::PxPhysics& physics_;
    std::map<std::string, MaterialProps> props_;
    std::map<std::string, PxPtr<physx::PxMaterial>> materials_;
};

}  // namespace p2s
