// Turns p2s shapes into PhysX shapes on an actor. Convex hulls are cooked here and written back into the
// p2s shape (vertices, planes, triangles), so the checker and the renderer see exactly what PhysX collides.
#pragma once

#include <vector>

#include "core/Shape.h"
#include "physx/Materials.h"
#include "physx/Runtime.h"

namespace p2s {

struct CollisionFilter {
    uint32_t group = 1;     // this shape's group bit
    uint32_t ignores = 0;   // groups it never collides with
    uint32_t entity = 0;    // owning entity index, for contact reports
    uint32_t assemblyId = 0;  // shapes with the same nonzero assembly never collide (a vehicle and its parts)

    physx::PxFilterData data() const { return physx::PxFilterData(group, ignores, entity, assemblyId); }
};

class ShapeFactory {
public:
    ShapeFactory(Runtime& runtime, const MaterialLibrary& materials) : runtime_(runtime), materials_(materials) {}

    // Creates one exclusive PhysX shape per p2s shape on `actor`. `shapes` may be updated (cooked hulls).
    void attach(physx::PxRigidActor& actor, std::vector<Shape>& shapes, const CollisionFilter& filter,
                const std::string& owner, double contactOffset = 0.02) const;
    // Shapes made from a mesh take their own reference, so the returned one may be dropped after attaching.
    // Cooking failures are reported as scene errors against `owner`.
    PxPtr<physx::PxConvexMesh> cookConvex(const std::vector<Vec3>& points, const std::string& owner) const;

private:
    PxPtr<physx::PxTriangleMesh> cookTriangleMesh(const Shape& mesh, const std::string& owner) const;

    Runtime& runtime_;
    const MaterialLibrary& materials_;
};

// Sets the filter words of every shape on an actor (used by the world's filter shader and contact reports).
void setFilter(physx::PxRigidActor& actor, const CollisionFilter& filter);

}  // namespace p2s
