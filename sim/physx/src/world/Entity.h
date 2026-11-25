// Base class of everything in a scene: bodies, vehicles, deformables, particle systems, signals.
#pragma once

#include <span>
#include <string>
#include <vector>

#include "core/Recorder.h"
#include "core/Shape.h"
#include "physx/Px.h"
#include "world/Events.h"

namespace p2s {

class World;

struct Energy {
    double kinetic = 0;    // J, translational + rotational
    double potential = 0;  // J, gravitational, relative to z = 0
};

class Entity {
public:
    Entity(std::string id, std::string kind) : id_(std::move(id)), kind_(std::move(kind)) {}
    virtual ~Entity() = default;
    Entity(const Entity&) = delete;
    Entity& operator=(const Entity&) = delete;

    const std::string& id() const { return id_; }
    const std::string& kind() const { return kind_; }

    // Resolve references to other entities, once every entity exists.
    virtual void link(World&) {}
    // Controllers and external forces, before each physics step.
    virtual void beforeStep(World&, double /*dt*/) {}
    // Read back state, detect events, after each physics step.
    virtual void afterStep(World&, double /*dt*/) {}
    // Per-frame output streams.
    virtual void record(World&, Recorder&) = 0;
    // Static description for x.json: kind, geometry, parameters. Called after the run.
    virtual Json describe() const = 0;

    // Actions that triggers may send ("brake", "release", "start", ...).
    virtual std::vector<std::string> verbs() const { return {}; }
    virtual void act(World&, const Action& a) { throw SpecError(id_ + ": no action '" + a.verb + "'"); }

    virtual Energy energy(const World&) const { return {}; }

    // Rigid entities expose their PhysX actor (for joints, attachments, contacts) and their shapes at the
    // current pose (for distance queries by deformables and particles).
    virtual bool isRigid() const { return false; }
    virtual physx::PxRigidActor& actor() { throw SpecError(id_ + " is not a rigid body"); }
    virtual const std::vector<Shape>& shapes() const { static const std::vector<Shape> none; return none; }
    virtual Pose pose() const { return {}; }
    virtual Vec3 velocity() const { return {}; }

    // Deformables and particles: their current surface points in world space, for contact detection that
    // PhysX's rigid contact reports do not cover. A deformable mounted on a host also reports as the host.
    virtual std::span<const Vec3> contactPoints() const { return {}; }
    // How far those points stay from a surface they rest on (particle radius, cloth thickness).
    virtual double contactRadius() const { return 0; }
    virtual const std::string& contactAlias() const { static const std::string none; return none; }

private:
    std::string id_, kind_;
};

// World-space bounds of shapes placed at `pose` (each shape's own box, then merged).
inline Aabb boundsOf(const std::vector<Shape>& shapes, const Pose& pose = {}) {
    Aabb box;
    for (const Shape& s : shapes) {
        const Aabb b = s.bounds().transformed(pose * s.local);
        if (b.empty()) continue;
        box.add(b.lo);
        box.add(b.hi);
    }
    return box;
}

// The "body/<id>/pose" row: position, then quaternion xyzw.
inline void putPose(Recorder& rec, const std::string& key, const Pose& p) {
    rec.put(key, {float(p.p.x), float(p.p.y), float(p.p.z), float(p.q.x), float(p.q.y), float(p.q.z), float(p.q.w)});
}

}  // namespace p2s
