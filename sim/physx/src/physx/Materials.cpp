#include "physx/Materials.h"

#include <stdexcept>

using namespace physx;

namespace p2s {

namespace {

const std::vector<std::string> kCombineModes = {"average", "min", "max", "multiply"};

PxCombineMode::Enum combineMode(const std::string& name) {
    if (name == "min") return PxCombineMode::eMIN;
    if (name == "max") return PxCombineMode::eMAX;
    if (name == "multiply") return PxCombineMode::eMULTIPLY;
    return PxCombineMode::eAVERAGE;
}

std::vector<std::string> presetNames() {
    std::vector<std::string> names;
    for (const auto& [name, _] : MaterialLibrary::presets()) names.push_back(name);
    return names;
}

}  // namespace

const std::map<std::string, MaterialProps>& MaterialLibrary::presets() {
    // Coefficients against a like surface; densities of the bulk material.
    static const std::map<std::string, MaterialProps> table = {
        {"default", {0.5, 0.4, 0.1, 1000}},
        {"steel", {0.5, 0.4, 0.05, 7850}},
        {"aluminium", {0.6, 0.45, 0.1, 2700}},
        {"rubber", {0.9, 0.8, 0.6, 1100}},
        {"tyre", {0.9, 0.8, 0.0, 1100}},
        {"wood", {0.5, 0.35, 0.3, 600}},
        {"concrete", {0.8, 0.7, 0.05, 2400}},
        {"asphalt", {0.8, 0.7, 0.0, 2300}},
        {"ice", {0.05, 0.03, 0.1, 917}},
        {"glass", {0.6, 0.4, 0.4, 2500}},
        {"plastic", {0.4, 0.3, 0.4, 950}},
        {"ceramic", {0.6, 0.5, 0.3, 2400}},
        {"flesh", {0.8, 0.7, 0.0, 1000}},
        {"track", {0.02, 0.02, 0.0, 7850, "min", "average"}},  // low-friction tracked-vehicle boxes; traction is a controller
        {"cloth", {0.5, 0.4, 0.0, 300}},
    };
    return table;
}

MaterialLibrary::MaterialLibrary(PxPhysics& physics) : physics_(physics) {
    for (const auto& [id, p] : presets()) create(id, p);
}

void MaterialLibrary::create(const std::string& id, const MaterialProps& p) {
    PxPtr<PxMaterial> m(
        physics_.createMaterial(float(p.staticFriction), float(p.dynamicFriction), float(p.restitution)));
    if (!m) throw std::runtime_error("createMaterial failed for '" + id + "'");
    m->setFrictionCombineMode(combineMode(p.frictionCombine));
    m->setRestitutionCombineMode(combineMode(p.restitutionCombine));
    props_[id] = p;
    materials_[id] = std::move(m);
}

void MaterialLibrary::add(const Spec& s) {
    const std::string id = s.str("id");
    if (!userDefined_.insert(id).second) s["id"].fail("material '" + id + "' is defined twice");
    MaterialProps p = props(s.oneOf("preset", presetNames(), "default"));
    p.staticFriction = s.num("static_friction", p.staticFriction);
    p.dynamicFriction = s.num("dynamic_friction", std::min(p.dynamicFriction, p.staticFriction));
    p.restitution = s.num("restitution", p.restitution);
    p.density = s.positive("density", p.density);
    p.frictionCombine = s.oneOf("friction_combine", kCombineModes, p.frictionCombine);
    p.restitutionCombine = s.oneOf("restitution_combine", kCombineModes, p.restitutionCombine);
    if (p.staticFriction < 0 || p.dynamicFriction < 0) s.fail("friction must be >= 0");
    if (p.restitution < 0 || p.restitution > 1) s.fail("restitution must be in [0, 1]");
    create(id, p);
}

PxMaterial& MaterialLibrary::get(const std::string& id, const std::string& usedBy) const {
    const auto it = materials_.find(id);
    if (it == materials_.end()) throw SpecError(usedBy + ": unknown material '" + id + "' (define it in materials)");
    return *it->second;
}

const MaterialProps& MaterialLibrary::props(const std::string& id) const {
    const auto it = props_.find(id);
    if (it == props_.end()) throw SpecError("unknown material '" + id + "'");
    return it->second;
}

Json MaterialLibrary::toJson() const {
    Json out = Json::object();
    for (const auto& [id, p] : props_)
        out[id] = {{"static_friction", p.staticFriction}, {"dynamic_friction", p.dynamicFriction},
                   {"restitution", p.restitution}, {"density", p.density},
                   {"friction_combine", p.frictionCombine}, {"restitution_combine", p.restitutionCombine}};
    return out;
}

}  // namespace p2s
