// Collision shapes as analytic implicit surfaces: the one 3D model that the simulator collides, the checker
// measures (penetration, distance, camera visibility) and Blender draws. No PhysX dependency.
//
// Shape types (local frame; capsules and cylinders run along local +Z):
//   box {half_extents}           sphere {radius}            capsule {radius, half_length}
//   cylinder {radius, half_length}                          prism {profile [[x, y], ...] (convex, CCW), z0, z1}
//   convex {planes, vertices}    (from a point cloud; the simulator fills in the hull)
//   mesh {vertices, triangles}   (static bodies only; signed distance uses the nearest triangle's normal)
// A "container" (open-topped hollow cylinder: cups, glasses, tanks) expands into a ring of boxes plus a base.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "core/Math.h"
#include "core/Spec.h"

namespace p2s {

struct Plane {
    Vec3 n;      // outward unit normal
    double d;    // n . x + d = 0 on the plane; positive outside
    double distance(const Vec3& x) const { return n.dot(x) + d; }
};

enum class ShapeType { Box, Sphere, Capsule, Cylinder, Prism, Convex, Mesh };

class Shape {
public:
    ShapeType type = ShapeType::Box;
    Pose local;                       // pose in the owning body's frame
    std::string material = "default";
    Vec3 halfExtents{0.5, 0.5, 0.5};  // box
    double radius = 0.5;              // sphere, capsule, cylinder
    double halfLength = 0.5;          // capsule, cylinder (along local Z, excluding caps)
    std::vector<Vec3> vertices;       // prism (both caps), convex hull, mesh
    std::vector<std::array<int, 3>> triangles;  // mesh (and the triangulated surface of every polyhedron)
    std::vector<Plane> planes;        // prism, convex: the polyhedron's faces

    // Parses one entry of a body's "shapes" list. A container returns several boxes.
    static std::vector<Shape> parse(const Spec& s);
    static Shape box(const Vec3& halfExtents, const Pose& local = {});
    static Shape sphere(double r, const Pose& local = {});
    static Shape capsule(double r, double halfLength, const Pose& local = {});
    static Shape cylinder(double r, double halfLength, const Pose& local = {});
    static Shape prism(const std::vector<std::array<double, 2>>& profile, double z0, double z1, const Pose& local = {});
    // The hull's planes and surface triangles are filled in when PhysX cooks it (ShapeFactory); until then sdf(),
    // raycast() and volume() have nothing to work with.
    static Shape convex(std::vector<Vec3> points, const Pose& local = {});
    static Shape mesh(std::vector<Vec3> vertices, std::vector<std::array<int, 3>> triangles, const Pose& local = {});

    std::string typeName() const;  // "box", "sphere", ... as in the scene and in X

    // All in the shape's own frame (apply `local` and the body pose to get world quantities).
    double sdf(const Vec3& p) const;
    std::optional<double> raycast(const Vec3& origin, const Vec3& dir, double maxT) const;  // dir unit length
    std::vector<Vec3> surfaceSamples() const;  // points on the surface: corners, edges, faces
    Aabb bounds() const;
    double volume() const;

    Json toJson() const;
    static Shape fromJson(const Json& j);  // inverse of toJson (used by the checker and Blender)

private:
    void buildPrismSurface();  // prism `vertices` (bottom cap, then top cap) -> planes and triangles
};

// A rigid set of shapes at a pose: what the checker needs to ask "how far is this point inside the body?".
struct PosedShapes {
    const std::vector<Shape>& shapes;
    Pose pose;

    double sdf(const Vec3& world) const;
    std::optional<double> raycast(const Vec3& origin, const Vec3& dir, double maxT) const;
};

}  // namespace p2s
