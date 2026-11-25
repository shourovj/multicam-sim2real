#include "physx/ShapeFactory.h"

#include <stdexcept>

using namespace physx;

namespace p2s {

namespace {

// PhysX capsules run along local X; p2s capsules and cylinders along local Z.
const Quat kXtoZ = Quat::axisAngle({0, 1, 0}, -kPi / 2);

std::vector<Vec3> cylinderPoints(double r, double hl, int segments) {
    std::vector<Vec3> pts;
    for (int i = 0; i < segments; ++i) {
        const double a = 2 * kPi * i / segments;
        pts.push_back({r * std::cos(a), r * std::sin(a), -hl});
        pts.push_back({r * std::cos(a), r * std::sin(a), hl});
    }
    return pts;
}

}  // namespace

void setFilter(PxRigidActor& actor, const CollisionFilter& f) {
    std::vector<PxShape*> shapes(actor.getNbShapes());
    actor.getShapes(shapes.data(), PxU32(shapes.size()));
    for (PxShape* s : shapes) {
        s->setSimulationFilterData(f.data());
        s->setQueryFilterData(f.data());
    }
}

PxPtr<PxConvexMesh> ShapeFactory::cookConvex(const std::vector<Vec3>& points, const std::string& owner) const {
    std::vector<PxVec3> pts;
    for (const Vec3& p : points) pts.push_back(px(p));
    PxConvexMeshDesc desc;
    desc.points.count = PxU32(pts.size());
    desc.points.stride = sizeof(PxVec3);
    desc.points.data = pts.data();
    desc.flags = PxConvexFlag::eCOMPUTE_CONVEX;
    desc.vertexLimit = 255;
    PxPtr<PxConvexMesh> mesh(
        PxCreateConvexMesh(runtime_.cookingParams(), desc, runtime_.physics().getPhysicsInsertionCallback()));
    if (!mesh) throw SpecError(owner + ": convex hull cooking failed (degenerate or flat point set?)");
    return mesh;
}

PxPtr<PxTriangleMesh> ShapeFactory::cookTriangleMesh(const Shape& s, const std::string& owner) const {
    std::vector<PxVec3> v;
    std::vector<PxU32> t;
    for (const Vec3& p : s.vertices) v.push_back(px(p));
    for (const auto& tri : s.triangles) t.insert(t.end(), {PxU32(tri[0]), PxU32(tri[1]), PxU32(tri[2])});
    PxTriangleMeshDesc d;
    d.points.count = PxU32(v.size());
    d.points.stride = sizeof(PxVec3);
    d.points.data = v.data();
    d.triangles.count = PxU32(s.triangles.size());
    d.triangles.stride = 3 * sizeof(PxU32);
    d.triangles.data = t.data();
    PxPtr<PxTriangleMesh> mesh(
        PxCreateTriangleMesh(runtime_.cookingParams(), d, runtime_.physics().getPhysicsInsertionCallback()));
    if (!mesh) throw SpecError(owner + ": triangle mesh cooking failed");
    return mesh;
}

namespace {

// Copies the cooked hull back into the p2s shape, so the checker and renderer see the hull PhysX collides.
void readHull(const PxConvexMesh& mesh, Shape& shape) {
    const PxVec3* v = mesh.getVertices();
    const PxU8* idx = mesh.getIndexBuffer();
    shape.vertices.clear();
    shape.planes.clear();
    shape.triangles.clear();
    for (PxU32 i = 0; i < mesh.getNbVertices(); ++i) shape.vertices.push_back(fromPx(v[i]));
    for (PxU32 i = 0; i < mesh.getNbPolygons(); ++i) {
        PxHullPolygon poly;
        mesh.getPolygonData(i, poly);
        shape.planes.push_back({{poly.mPlane[0], poly.mPlane[1], poly.mPlane[2]}, poly.mPlane[3]});
        for (PxU16 k = 1; k + 1 < poly.mNbVerts; ++k)
            shape.triangles.push_back({idx[poly.mIndexBase], idx[poly.mIndexBase + k], idx[poly.mIndexBase + k + 1]});
    }
}

}  // namespace

void ShapeFactory::attach(PxRigidActor& actor, std::vector<Shape>& shapes, const CollisionFilter& filter,
                          const std::string& owner, double contactOffset) const {
    for (Shape& s : shapes) {
        PxMaterial& mat = materials_.get(s.material, owner);
        Pose local = s.local;
        PxShape* shape = nullptr;
        switch (s.type) {
            case ShapeType::Box:
                shape = PxRigidActorExt::createExclusiveShape(actor, PxBoxGeometry(px(s.halfExtents)), mat);
                break;
            case ShapeType::Sphere:
                shape = PxRigidActorExt::createExclusiveShape(actor, PxSphereGeometry(float(s.radius)), mat);
                break;
            case ShapeType::Capsule:
                shape = PxRigidActorExt::createExclusiveShape(
                    actor, PxCapsuleGeometry(float(s.radius), float(s.halfLength)), mat);
                local.q = local.q * kXtoZ;
                break;
            case ShapeType::Cylinder:
            case ShapeType::Prism:
            case ShapeType::Convex: {
                const auto mesh = cookConvex(
                    s.type == ShapeType::Cylinder ? cylinderPoints(s.radius, s.halfLength, 32) : s.vertices, owner);
                shape = PxRigidActorExt::createExclusiveShape(actor, PxConvexMeshGeometry(mesh.get()), mat);
                if (s.type == ShapeType::Convex) readHull(*mesh, s);
                break;
            }
            case ShapeType::Mesh: {
                if (actor.getConcreteType() != PxConcreteType::eRIGID_STATIC)
                    throw SpecError(owner + ": mesh shapes are only allowed on static bodies; use convex or prism");
                const auto mesh = cookTriangleMesh(s, owner);
                shape = PxRigidActorExt::createExclusiveShape(actor, PxTriangleMeshGeometry(mesh.get()), mat);
                break;
            }
        }
        if (!shape) throw SpecError(owner + ": PhysX rejected a " + s.typeName() + " shape");
        shape->setLocalPose(px(local));
        shape->setContactOffset(float(contactOffset));
        shape->setRestOffset(0.0f);
    }
    setFilter(actor, filter);
}

}  // namespace p2s
