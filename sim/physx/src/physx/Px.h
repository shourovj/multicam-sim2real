// PhysX glue: ownership of PhysX objects and conversions between p2s and PhysX math types.
#pragma once

#include <memory>

#include "PxPhysicsAPI.h"
#include "core/Math.h"

namespace p2s {

// PhysX objects are freed with release(), never delete: unique_ptr with a releasing deleter.
struct PxReleaser {
    template <class T>
    void operator()(T* p) const {
        if (p) p->release();
    }
};
template <class T>
using PxPtr = std::unique_ptr<T, PxReleaser>;

inline physx::PxVec3 px(const Vec3& v) { return {float(v.x), float(v.y), float(v.z)}; }
inline physx::PxQuat px(const Quat& q) { return physx::PxQuat(float(q.x), float(q.y), float(q.z), float(q.w)); }
inline physx::PxTransform px(const Pose& p) { return physx::PxTransform(px(p.p), px(p.q)); }
inline Vec3 fromPx(const physx::PxVec3& v) { return {v.x, v.y, v.z}; }
inline Quat fromPx(const physx::PxQuat& q) { return {q.x, q.y, q.z, q.w}; }
inline Pose fromPx(const physx::PxTransform& t) { return {fromPx(t.p), fromPx(t.q)}; }

}  // namespace p2s
