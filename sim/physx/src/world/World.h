// A simulated scene: the PhysX scene, every entity, the clock, events, triggers, contacts and fields.
// Builds itself from the scene JSON (S), runs, and writes the physics output X.
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "core/Actuators.h"
#include "core/Wind.h"
#include "physx/Materials.h"
#include "physx/ShapeFactory.h"
#include "world/Contacts.h"
#include "world/Entity.h"

namespace p2s {

class Camera;

struct Clock {
    double fps = 24;
    int frames = 1;
    int substeps = 20;       // physics steps per frame
    double prerollS = 0;     // simulated before t = 0, not recorded (cloth settling, vehicles reaching speed)
    double dt() const { return 1.0 / (fps * substeps); }
};

// Smaller steps while two bodies are about to hit or are in contact (the tank-truck "crash window").
// Active from when the two first come within within_m until hold_s after they first touch (or after they first
// came close, if they never touch): the approach and the impact, not the long contact afterwards.
struct Refinement {
    std::string a, b;
    double withinM = 1.0, holdS = 0.4;
    int factor = 8;
    std::optional<double> closeAt;
};

class World {
public:
    using Factory = std::function<std::unique_ptr<Entity>(const Spec&, World&)>;
    // Scene keys that hold lists of entities and how to build each entry, in build order.
    using FactoryList = std::vector<std::pair<std::string, Factory>>;
    static const FactoryList& factories();

    World(Runtime& runtime, const Spec& scene);
    ~World();

    // Simulates every frame and writes X to outDir (x.json + streams/).
    void run(const std::string& outDir, const Json& runInfo);

    physx::PxScene& scene() { return *scene_; }
    physx::PxPhysics& physics() { return runtime_.physics(); }
    Runtime& runtime() { return runtime_; }
    const MaterialLibrary& materials() const { return materials_; }
    const ShapeFactory& shapeFactory() const { return shapeFactory_; }
    const Clock& clock() const { return clock_; }
    double time() const { return t_; }
    Vec3 gravity() const { return gravity_; }
    EventLog& events() { return events_; }
    ActuatorLog& actuators() { return actuators_; }
    const WindField& wind() const { return wind_; }
    std::mt19937& rng() { return rng_; }

    Entity& entity(const std::string& id, const std::string& usedBy);
    template <class T>
    T& entityAs(const std::string& id, const std::string& usedBy) {
        Entity& e = entity(id, usedBy);
        auto* t = dynamic_cast<T*>(&e);
        if (!t) throw SpecError(usedBy + ": '" + id + "' is a " + e.kind() + ", which cannot be used here");
        return *t;
    }
    const std::vector<std::unique_ptr<Entity>>& entities() const { return entities_; }
    // Collision filter for a new entity. Entities in the same nonzero assembly never collide with each other.
    CollisionFilter filterFor(const std::string& entityId, const std::string& assembly = "", uint32_t category = 1,
                              uint32_t ignores = 0);
    std::string entityName(uint32_t index) const;
    // The assembly of a rigid entity, creating one named after it (and re-filtering its shapes) if it has none, so
    // parts mounted on it (a crumple zone, a flag) never collide with it.
    std::string joinAssembly(Entity& host);

private:
    void build(const Spec& scene);
    void releaseEntities();
    void step(double h);
    void detectDeformableContacts();
    // Whether any sampled point of `a` is closer than `touch` to a sampled point of `b`.
    bool pointsTouch(std::span<const Vec3> a, const Aabb& boxA, std::span<const Vec3> b, const Aabb& boxB,
                     double touch);
    void recordFrame(int frame, Recorder& rec);
    bool refining();
    Json describe() const;

    Runtime& runtime_;
    MaterialLibrary materials_;
    ShapeFactory shapeFactory_;
    EventLog events_;
    ContactReporter contacts_;
    PxPtr<physx::PxScene> scene_;
    std::vector<std::unique_ptr<Entity>> entities_;
    std::map<std::string, size_t> index_;
    std::map<std::string, uint32_t> assemblies_;
    std::map<std::string, std::string> assemblyOf_;  // entity id -> assembly name
    std::vector<uint32_t> assemblyIds_;              // per entity, 0 = none
    std::vector<Aabb> pointBoxes_;                   // contact-point bounds per entity, this step
    std::vector<std::pair<uint64_t, Vec3>> cells_;   // spatial hash scratch for pointsTouch
    std::vector<std::unique_ptr<Camera>> cameras_;
    std::vector<Trigger> triggers_;
    std::vector<Refinement> refinements_;
    ActuatorLog actuators_;
    WindField wind_;
    Clock clock_;
    Vec3 gravity_{0, 0, -9.81};
    double t_ = 0;
    long steps_ = 0, fineSteps_ = 0;
    std::mt19937 rng_;
    Json sceneInfo_;
    std::vector<Json> energy_;
};

// Every built-in entity kind, in build order (bodies/Registry.cpp).
World::FactoryList builtinEntities();

}  // namespace p2s
