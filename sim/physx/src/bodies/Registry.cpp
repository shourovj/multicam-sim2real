// Every scene list key and the entity class that builds its entries, in build order: things that others
// refer to (bodies) come before things that refer to them (joints, attachments, controllers).
#include "bodies/Cloth.h"
#include "bodies/Joint.h"
#include "bodies/Multirotor.h"
#include "bodies/Particles.h"
#include "bodies/RigidBody.h"
#include "bodies/Signal.h"
#include "bodies/SoftBody.h"
#include "bodies/Vehicle.h"
#include "bodies/Walker.h"

namespace p2s {

namespace {

template <class T>
World::Factory make() {
    return [](const Spec& s, World& w) { return std::make_unique<T>(s, w); };
}

}  // namespace

World::FactoryList builtinEntities() {
    return {
        {"bodies", make<RigidBody>()},
        {"multirotors", make<Multirotor>()},
        {"walkers", make<Walker>()},
        {"signals", make<TrafficSignal>()},
        {"vehicles", make<Vehicle>()},
        {"joints", make<Joint>()},
        {"soft_bodies", make<SoftBody>()},
        {"cloths", make<Cloth>()},
        {"particles", make<ParticleSystem>()},
    };
}

}  // namespace p2s
