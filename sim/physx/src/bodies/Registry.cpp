// Every scene list key and the entity class that builds its entries, in build order: things that others
// refer to (bodies) come before things that refer to them (joints, attachments, controllers).
#include "bodies/RigidBody.h"

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
    };
}

}  // namespace p2s
