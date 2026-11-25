#include "core/Shape.h"

#include <span>
#include <stdexcept>

namespace p2s {

namespace {

constexpr double kTraceEps = 1e-4;
constexpr int kTraceSteps = 128;
constexpr std::array<const char*, 7> kTypeNames = {"box", "sphere", "capsule", "cylinder", "prism", "convex", "mesh"};

double length2(double x, double y) { return std::sqrt(x * x + y * y); }

// Signed distance to a convex CCW polygon in the plane (z of the vertices is ignored).
double polygonSdf(std::span<const Vec3> poly, double x, double y) {
    const size_t n = poly.size();
    double inside = -1e300, outside = 1e300;
    bool isInside = true;
    for (size_t i = 0; i < n; ++i) {
        const Vec3& a = poly[i];
        const Vec3& b = poly[(i + 1) % n];
        const double ex = b.x - a.x, ey = b.y - a.y, len = length2(ex, ey);
        const double nx = ey / len, ny = -ex / len;  // outward for CCW
        const double side = (x - a.x) * nx + (y - a.y) * ny;
        inside = std::max(inside, side);
        if (side > 0) isInside = false;
        const double u = std::clamp(((x - a.x) * ex + (y - a.y) * ey) / (len * len), 0.0, 1.0);
        outside = std::min(outside, length2(x - (a.x + u * ex), y - (a.y + u * ey)));
    }
    return isInside ? inside : outside;
}

// Combines a 2D cross-section distance with a slab |z| distance, as for an extrusion.
double extrude(double d2, double dz) {
    return std::min(std::max(d2, dz), 0.0) + length2(std::max(d2, 0.0), std::max(dz, 0.0));
}

double triangleDistance(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c) {
    // Closest point on triangle (Ericson, Real-Time Collision Detection 5.1.5).
    const Vec3 ab = b - a, ac = c - a, ap = p - a;
    const double d1 = ab.dot(ap), d2 = ac.dot(ap);
    if (d1 <= 0 && d2 <= 0) return (p - a).norm();
    const Vec3 bp = p - b;
    const double d3 = ab.dot(bp), d4 = ac.dot(bp);
    if (d3 >= 0 && d4 <= d3) return (p - b).norm();
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) return (p - (a + ab * (d1 / (d1 - d3)))).norm();
    const Vec3 cp = p - c;
    const double d5 = ab.dot(cp), d6 = ac.dot(cp);
    if (d6 >= 0 && d5 <= d6) return (p - c).norm();
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) return (p - (a + ac * (d2 / (d2 - d6)))).norm();
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
        return (p - (b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))))).norm();
    const double denom = 1.0 / (va + vb + vc);
    return (p - (a + ab * (vb * denom) + ac * (vc * denom))).norm();
}

std::optional<double> rayTriangle(const Vec3& o, const Vec3& d, const Vec3& a, const Vec3& b, const Vec3& c) {
    const Vec3 e1 = b - a, e2 = c - a, h = d.cross(e2);
    const double det = e1.dot(h);
    if (std::fabs(det) < 1e-12) return std::nullopt;
    const Vec3 s = o - a;
    const double u = s.dot(h) / det;
    if (u < 0 || u > 1) return std::nullopt;
    const Vec3 q = s.cross(e1);
    const double v = d.dot(q) / det;
    if (v < 0 || u + v > 1) return std::nullopt;
    const double t = e2.dot(q) / det;
    return t > 0 ? std::optional<double>(t) : std::nullopt;
}

std::vector<Vec3> fibonacciSphere(int n, double r) {
    std::vector<Vec3> out;
    const double golden = kPi * (3.0 - std::sqrt(5.0));
    for (int i = 0; i < n; ++i) {
        const double z = 1.0 - 2.0 * (i + 0.5) / n, rr = std::sqrt(1.0 - z * z), th = golden * i;
        out.push_back(Vec3{rr * std::cos(th), rr * std::sin(th), z} * r);
    }
    return out;
}

std::vector<Vec3> ring(double r, double z, int n) {
    std::vector<Vec3> out;
    for (int i = 0; i < n; ++i) out.push_back({r * std::cos(2 * kPi * i / n), r * std::sin(2 * kPi * i / n), z});
    return out;
}

std::vector<Shape> container(const Spec& s, const Pose& local) {
    // Open-topped hollow cylinder, origin at the centre of the outside bottom.
    const double r = s.positive("radius"), h = s.positive("height");
    const double wall = s.positive("wall", 0.006), base = s.positive("base", 0.01);
    const int segments = s.integer("segments", 16);
    if (segments < 6 || segments > 256) s["segments"].fail("must be in 6..256");
    if (wall >= r) s["wall"].fail("must be less than the radius");
    if (base >= h) s["base"].fail("must be less than the height");
    std::vector<Shape> out;
    const double chord = 2.0 * r * std::sin(kPi / segments);
    for (int i = 0; i < segments; ++i) {
        const double a = 2 * kPi * (i + 0.5) / segments;
        const Pose seg{{(r - wall / 2) * std::cos(a), (r - wall / 2) * std::sin(a), h / 2},
                       Quat::axisAngle({0, 0, 1}, a)};
        out.push_back(Shape::box({wall / 2, chord / 2 + wall / 2, h / 2}, local * seg));
    }
    out.push_back(Shape::cylinder(r, base / 2, local * Pose{{0, 0, base / 2}, {}}));
    return out;
}

// A prism profile: at least three [x, y] points forming a convex, counter-clockwise, simple polygon.
std::vector<std::array<double, 2>> parseProfile(const Spec& prof) {
    if (!prof.isArray()) prof.fail("expected an array of [x, y]");
    std::vector<std::array<double, 2>> pts;
    for (size_t i = 0; i < prof.size(); ++i) {
        const Spec xy = prof.at(i);
        if (!xy.isArray() || xy.size() != 2) xy.fail("expected [x, y]");
        pts.push_back({xy.at(0).asNum(), xy.at(1).asNum()});
    }
    const size_t n = pts.size();
    if (n < 3) prof.fail("needs at least 3 points");
    // Every turn must be to the left, and the turns must add up to one full revolution (a pentagram also turns
    // left at every corner, but twice round).
    double turning = 0, area2 = 0;
    for (size_t i = 0; i < n; ++i) {
        const auto& a = pts[i];
        const auto& b = pts[(i + 1) % n];
        const auto& c = pts[(i + 2) % n];
        const double ex = b[0] - a[0], ey = b[1] - a[1], fx = c[0] - b[0], fy = c[1] - b[1];
        if (length2(ex, ey) < 1e-9) prof.at((i + 1) % n).fail("repeats the previous point");
        const double cross = ex * fy - ey * fx;
        if (cross < -1e-12)
            prof.fail("profile must be convex and counter-clockwise; split concave outlines into several prisms");
        turning += std::atan2(cross, ex * fx + ey * fy);
        area2 += a[0] * b[1] - b[0] * a[1];
    }
    if (area2 <= 1e-12) prof.fail("profile has no area");
    if (std::fabs(turning - 2 * kPi) > 1e-6) prof.fail("profile must be a simple convex polygon, wound once");
    return pts;
}

std::vector<std::array<int, 3>> parseTriangles(const Spec& tris, size_t vertexCount) {
    if (!tris.isArray()) tris.fail("expected an array of [i, j, k]");
    std::vector<std::array<int, 3>> out;
    for (size_t i = 0; i < tris.size(); ++i) {
        const Spec t = tris.at(i);
        if (!t.isArray() || t.size() != 3) t.fail("expected [i, j, k]");
        std::array<int, 3> tri{};
        for (int k = 0; k < 3; ++k) {
            const double v = t.at(k).asNum();
            if (v != std::floor(v) || v < 0 || v >= double(vertexCount)) t.at(k).fail("vertex index out of range");
            tri[k] = int(v);
        }
        out.push_back(tri);
    }
    return out;
}

}  // namespace

Shape Shape::box(const Vec3& he, const Pose& local) {
    Shape s;
    s.type = ShapeType::Box;
    s.halfExtents = he;
    s.local = local;
    return s;
}

Shape Shape::sphere(double r, const Pose& local) {
    Shape s;
    s.type = ShapeType::Sphere;
    s.radius = r;
    s.local = local;
    return s;
}

Shape Shape::capsule(double r, double hl, const Pose& local) {
    Shape s = sphere(r, local);
    s.type = ShapeType::Capsule;
    s.halfLength = hl;
    return s;
}

Shape Shape::cylinder(double r, double hl, const Pose& local) {
    Shape s = capsule(r, hl, local);
    s.type = ShapeType::Cylinder;
    return s;
}

Shape Shape::prism(const std::vector<std::array<double, 2>>& profile, double z0, double z1, const Pose& local) {
    Shape s;
    s.type = ShapeType::Prism;
    s.local = local;
    for (double z : {z0, z1})
        for (const auto& p : profile) s.vertices.push_back({p[0], p[1], z});
    s.buildPrismSurface();
    return s;
}

Shape Shape::convex(std::vector<Vec3> points, const Pose& local) {
    Shape s;
    s.type = ShapeType::Convex;
    s.local = local;
    s.vertices = std::move(points);
    return s;
}

Shape Shape::mesh(std::vector<Vec3> vertices, std::vector<std::array<int, 3>> triangles, const Pose& local) {
    Shape s;
    s.type = ShapeType::Mesh;
    s.local = local;
    s.vertices = std::move(vertices);
    s.triangles = std::move(triangles);
    return s;
}

void Shape::buildPrismSurface() {
    const int n = static_cast<int>(vertices.size() / 2);
    const double z0 = vertices.front().z, z1 = vertices.back().z;
    planes = {{{0, 0, -1}, z0}, {{0, 0, 1}, -z1}};
    triangles.clear();
    for (int k = 1; k + 1 < n; ++k) {
        triangles.push_back({0, k + 1, k});             // bottom cap, facing -z
        triangles.push_back({n, n + k, n + k + 1});     // top cap, facing +z
    }
    for (int i = 0; i < n; ++i) {
        const int j = (i + 1) % n;
        triangles.push_back({i, j, n + j});
        triangles.push_back({i, n + j, n + i});
        const Vec3 e = vertices[j] - vertices[i];
        const Vec3 nrm = Vec3{e.y, -e.x, 0}.normalized();
        planes.push_back({nrm, -nrm.dot(vertices[i])});
    }
}

std::vector<Shape> Shape::parse(const Spec& s) {
    const std::string type =
        s.oneOf("type", {"box", "sphere", "capsule", "cylinder", "prism", "convex", "mesh", "container"});
    const Pose local = s.pose();
    std::vector<Shape> out;
    if (type == "container") {
        out = container(s, local);
    } else if (type == "box") {
        const Vec3 he = s.has("size") ? s.vec3("size") * 0.5 : s.vec3("half_extents");
        if (he.x <= 0 || he.y <= 0 || he.z <= 0) s.fail("box extents must be > 0");
        out.push_back(box(he, local));
    } else if (type == "sphere") {
        out.push_back(sphere(s.positive("radius"), local));
    } else if (type == "capsule") {
        out.push_back(capsule(s.positive("radius"), s.positive("half_length"), local));
    } else if (type == "cylinder") {
        out.push_back(cylinder(s.positive("radius"), s.positive("half_length"), local));
    } else if (type == "prism") {
        const auto profile = parseProfile(s["profile"]);
        const double z0 = s.num("z0"), z1 = s.num("z1");
        if (z1 <= z0) s["z1"].fail("z1 must be > z0");
        out.push_back(prism(profile, z0, z1, local));
    } else if (type == "convex") {
        std::vector<Vec3> points = s.vec3s("points");
        if (points.size() < 4) s["points"].fail("needs at least 4 points");
        out.push_back(convex(std::move(points), local));
    } else {
        std::vector<Vec3> verts = s.vec3s("vertices");
        auto tris = parseTriangles(s["triangles"], verts.size());
        if (tris.empty()) s["triangles"].fail("needs at least one triangle");
        out.push_back(mesh(std::move(verts), std::move(tris), local));
    }
    const std::string material = s.str("material", "default");
    for (auto& sh : out) sh.material = material;
    return out;
}

double Shape::sdf(const Vec3& p) const {
    switch (type) {
        case ShapeType::Box: {
            const Vec3 q = p.cwiseAbs() - halfExtents;
            return q.cwiseMax({0, 0, 0}).norm() + std::min(q.maxCoeff(), 0.0);
        }
        case ShapeType::Sphere:
            return p.norm() - radius;
        case ShapeType::Capsule:
            return (p - Vec3{0, 0, std::clamp(p.z, -halfLength, halfLength)}).norm() - radius;
        case ShapeType::Cylinder:
            return extrude(length2(p.x, p.y) - radius, std::fabs(p.z) - halfLength);
        case ShapeType::Prism: {
            const std::span<const Vec3> cap(vertices.data(), vertices.size() / 2);
            const double z0 = vertices.front().z, z1 = vertices.back().z;
            return extrude(polygonSdf(cap, p.x, p.y), std::fabs(p.z - (z0 + z1) / 2) - (z1 - z0) / 2);
        }
        case ShapeType::Convex: {
            double d = -1e300;  // exact inside; a lower bound on the distance outside
            for (const auto& pl : planes) d = std::max(d, pl.distance(p));
            return d;
        }
        case ShapeType::Mesh: {
            double best = 1e300, sign = 1;
            for (const auto& t : triangles) {
                const Vec3 &a = vertices[t[0]], &b = vertices[t[1]], &c = vertices[t[2]];
                const double d = triangleDistance(p, a, b, c);
                if (d < best) {
                    best = d;
                    sign = (b - a).cross(c - a).dot(p - a) >= 0 ? 1 : -1;
                }
            }
            return best * sign;
        }
    }
    return 1e300;
}

std::optional<double> Shape::raycast(const Vec3& o, const Vec3& d, double maxT) const {
    if (type == ShapeType::Mesh) {
        std::optional<double> best;
        for (const auto& t : triangles)
            if (auto hit = rayTriangle(o, d, vertices[t[0]], vertices[t[1]], vertices[t[2]]); hit && *hit <= maxT)
                best = best ? std::min(*best, *hit) : *hit;
        return best;
    }
    // Sphere tracing: every distance above is exact or an under-estimate, so a step never passes the surface.
    double t = 0;
    for (int i = 0; i < kTraceSteps && t <= maxT; ++i) {
        const double dist = sdf(o + d * t);
        if (dist < kTraceEps) return t;
        t += dist;
    }
    return std::nullopt;
}

std::vector<Vec3> Shape::surfaceSamples() const {
    std::vector<Vec3> out;
    switch (type) {
        case ShapeType::Box: {
            const Vec3& h = halfExtents;
            for (int i = -1; i <= 1; ++i)
                for (int j = -1; j <= 1; ++j)
                    for (int k = -1; k <= 1; ++k)
                        if (i || j || k) out.push_back({i * h.x, j * h.y, k * h.z});  // corners, edges, faces
            break;
        }
        case ShapeType::Sphere:
            out = fibonacciSphere(48, radius);
            break;
        case ShapeType::Capsule:
            for (const Vec3& v : fibonacciSphere(48, radius))
                out.push_back(v + Vec3{0, 0, v.z >= 0 ? halfLength : -halfLength});
            for (double z : {-halfLength / 2, 0.0, halfLength / 2})
                for (const Vec3& v : ring(radius, z, 12)) out.push_back(v);
            break;
        case ShapeType::Cylinder:
            for (double z : {-halfLength, -halfLength / 2, 0.0, halfLength / 2, halfLength})
                for (const Vec3& v : ring(radius, z, 16)) out.push_back(v);
            for (double z : {-halfLength, halfLength}) {
                out.push_back({0, 0, z});
                for (const Vec3& v : ring(radius / 2, z, 8)) out.push_back(v);
            }
            break;
        default:
            out = vertices;
            for (const auto& t : triangles) out.push_back((vertices[t[0]] + vertices[t[1]] + vertices[t[2]]) / 3.0);
            break;
    }
    return out;
}

Aabb Shape::bounds() const {
    Aabb box;
    switch (type) {
        case ShapeType::Box: box.add(-halfExtents); box.add(halfExtents); break;
        case ShapeType::Sphere: box.add(Vec3{1, 1, 1} * -radius); box.add(Vec3{1, 1, 1} * radius); break;
        case ShapeType::Capsule:
        case ShapeType::Cylinder: {
            const double zr = type == ShapeType::Capsule ? halfLength + radius : halfLength;
            box.add({-radius, -radius, -zr});
            box.add({radius, radius, zr});
            break;
        }
        default:
            for (const Vec3& v : vertices) box.add(v);
    }
    return box;
}

double Shape::volume() const {
    switch (type) {
        case ShapeType::Box: return 8 * halfExtents.x * halfExtents.y * halfExtents.z;
        case ShapeType::Sphere: return 4.0 / 3.0 * kPi * radius * radius * radius;
        case ShapeType::Capsule: return kPi * radius * radius * (2 * halfLength + 4.0 / 3.0 * radius);
        case ShapeType::Cylinder: return kPi * radius * radius * 2 * halfLength;
        default: {
            double v = 0;  // divergence theorem over the closed triangulated surface
            for (const auto& t : triangles) v += vertices[t[0]].dot(vertices[t[1]].cross(vertices[t[2]])) / 6.0;
            return std::fabs(v);
        }
    }
}

std::string Shape::typeName() const { return kTypeNames[static_cast<size_t>(type)]; }

Json Shape::toJson() const {
    Json j = {{"type", typeName()},
              {"pos", p2s::toJson(local.p)},
              {"quat_xyzw", p2s::toJson(local.q)},
              {"material", material}};
    switch (type) {
        case ShapeType::Box: j["half_extents"] = p2s::toJson(halfExtents); break;
        case ShapeType::Sphere: j["radius"] = radius; break;
        case ShapeType::Capsule:
        case ShapeType::Cylinder: j["radius"] = radius; j["half_length"] = halfLength; break;
        default: {
            Json v = Json::array(), t = Json::array(), pl = Json::array();
            for (const Vec3& x : vertices) v.push_back(p2s::toJson(x));
            for (const auto& x : triangles) t.push_back({x[0], x[1], x[2]});
            for (const auto& x : planes) pl.push_back({x.n.x, x.n.y, x.n.z, x.d});
            j["vertices"] = v;
            j["triangles"] = t;
            if (!planes.empty()) j["planes"] = pl;
        }
    }
    return j;
}

Shape Shape::fromJson(const Json& j) {
    const auto v3 = [](const Json& a) { return Vec3{a[0].get<double>(), a[1].get<double>(), a[2].get<double>()}; };
    Shape s;
    const std::string type = j.at("type").get<std::string>();
    const auto it = std::find(kTypeNames.begin(), kTypeNames.end(), type);
    if (it == kTypeNames.end()) throw std::runtime_error("unknown shape type '" + type + "'");
    s.type = static_cast<ShapeType>(it - kTypeNames.begin());
    const auto& q = j.at("quat_xyzw");
    s.local = {v3(j.at("pos")), Quat{q[0].get<double>(), q[1].get<double>(), q[2].get<double>(), q[3].get<double>()}};
    s.material = j.value("material", "default");
    if (j.contains("half_extents")) s.halfExtents = v3(j["half_extents"]);
    s.radius = j.value("radius", 0.5);
    s.halfLength = j.value("half_length", 0.5);
    if (j.contains("vertices"))
        for (const auto& x : j["vertices"]) s.vertices.push_back(v3(x));
    if (j.contains("triangles"))
        for (const auto& x : j["triangles"]) s.triangles.push_back({x[0].get<int>(), x[1].get<int>(), x[2].get<int>()});
    if (j.contains("planes"))
        for (const auto& x : j["planes"]) s.planes.push_back({v3(x), x[3].get<double>()});
    return s;
}

double PosedShapes::sdf(const Vec3& world) const {
    double d = 1e300;
    for (const Shape& s : shapes) d = std::min(d, s.sdf((pose * s.local).transformInv(world)));
    return d;
}

std::optional<double> PosedShapes::raycast(const Vec3& origin, const Vec3& dir, double maxT) const {
    std::optional<double> best;
    for (const Shape& s : shapes) {
        const Pose w = pose * s.local;
        if (auto t = s.raycast(w.transformInv(origin), w.q.rotateInv(dir), maxT); t && (!best || *t < *best)) best = t;
    }
    return best;
}

}  // namespace p2s
