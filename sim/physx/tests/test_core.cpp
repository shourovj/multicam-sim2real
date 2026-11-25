// Core (no GPU): geometry, the spec reader, cameras, recorder.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "third_party/doctest.h"

#include "core/Camera.h"
#include "core/Shape.h"
#include "core/Spec.h"

using namespace p2s;

namespace {

Shape parseOne(const Json& j, bool concave = false) { return Shape::parse(Spec::fromJson(j, "shape"), concave).front(); }

double meshVolume(const TriMesh& m) {
    double v = 0;  // divergence theorem: positive only if every triangle winds outward
    for (const auto& t : m.triangles) v += m.vertices[size_t(t[0])].dot(m.vertices[size_t(t[1])].cross(m.vertices[size_t(t[2])])) / 6;
    return v;
}

}  // namespace

TEST_CASE("signed distances of primitives") {
    const Shape box = Shape::box({1, 2, 3});
    CHECK(box.sdf({0, 0, 0}) == doctest::Approx(-1));
    CHECK(box.sdf({2, 0, 0}) == doctest::Approx(1));
    CHECK(box.sdf({2, 3, 0}) == doctest::Approx(std::sqrt(2.0)));
    CHECK(Shape::sphere(0.5).sdf({1, 0, 0}) == doctest::Approx(0.5));
    CHECK(Shape::capsule(0.5, 1).sdf({0, 0, 2}) == doctest::Approx(0.5));
    CHECK(Shape::cylinder(1, 1).sdf({0, 0, 0}) == doctest::Approx(-1));
    CHECK(Shape::cylinder(1, 1).sdf({2, 0, 0}) == doctest::Approx(1));
}

TEST_CASE("prisms: convex for rigid bodies, concave only for soft bodies") {
    const Json l = {{"type", "prism"}, {"z0", -1}, {"z1", 1},
                    {"profile", {{0, 0}, {2, 0}, {2, 1}, {1, 1}, {1, 2}, {0, 2}}}};  // an L
    CHECK_THROWS_AS(parseOne(l), SpecError);
    const Shape s = parseOne(l, true);
    CHECK(s.sdf({0.5, 0.5, 0}) < 0);
    CHECK(s.sdf({1.5, 1.5, 0}) == doctest::Approx(0.5));  // in the notch: 0.5 from both inner edges
    CHECK(meshVolume(s.surfaceMesh()) == doctest::Approx(3 * 2.0).epsilon(1e-9));
    const Json cw = {{"type", "prism"}, {"z0", 0}, {"z1", 1}, {"profile", {{0, 0}, {0, 1}, {1, 1}, {1, 0}}}};
    CHECK_THROWS_AS(parseOne(cw, true), SpecError);  // clockwise
}

TEST_CASE("surface meshes are closed, wind outward and enclose the right volume") {
    const double pi = kPi;
    CHECK(meshVolume(Shape::box({1, 2, 3}).surfaceMesh()) == doctest::Approx(48));
    CHECK(meshVolume(Shape::cylinder(1, 1).surfaceMesh(64)) == doctest::Approx(2 * pi).epsilon(0.01));
    CHECK(meshVolume(Shape::sphere(1).surfaceMesh(64)) == doctest::Approx(4.0 / 3 * pi).epsilon(0.02));
    CHECK(meshVolume(Shape::capsule(0.5, 1).surfaceMesh(64)) == doctest::Approx(Shape::capsule(0.5, 1).volume()).epsilon(0.02));
}

TEST_CASE("raycasts hit primitives where expected") {
    const auto hit = Shape::box({1, 1, 1}).raycast({-5, 0, 0}, {1, 0, 0}, 100);
    REQUIRE(hit);
    CHECK(*hit == doctest::Approx(4).epsilon(1e-3));
    CHECK_FALSE(Shape::sphere(1).raycast({-5, 3, 0}, {1, 0, 0}, 100));
}

TEST_CASE("shapes round-trip through X") {
    const Shape s = parseOne({{"type", "capsule"}, {"radius", 0.2}, {"half_length", 0.7}, {"pos", {1, 2, 3}}, {"yaw_deg", 30}});
    const Shape r = Shape::fromJson(s.toJson());
    CHECK(r.sdf({1, 0, 0}) == doctest::Approx(s.sdf({1, 0, 0})));
    CHECK(r.local.p.y == doctest::Approx(2));
}

TEST_CASE("the spec reader names the path of an error and rejects typos") {
    const Spec s = Spec::fromJson({{"bodies", {{{"id", "a"}, {"masss", 3}}}}});
    const Spec body = s.items("bodies").front();
    CHECK(body.str("id") == "a");
    CHECK_THROWS_WITH_AS(s.rejectUnknownKeys(), doctest::Contains("scene.bodies[0].masss"), SpecError);
    CHECK_THROWS_WITH_AS(body.positive("radius"), doctest::Contains("scene.bodies[0]"), SpecError);
}

TEST_CASE("cameras: Blender convention, projection, attached cameras look along the parent's +X") {
    const Spec fixed = Spec::fromJson({{"id", "c"}, {"type", "fixed"}, {"position", {0, -10, 0}}, {"look_at", {0, 0, 0}}});
    Camera cam(fixed);
    const Pose p = cam.update(0, [](const std::string&) { return Pose{}; });
    const Projection centre = project(p, cam.intrinsics(), {0, 0, 0});
    CHECK(centre.u == doctest::Approx(640));
    CHECK(centre.v == doctest::Approx(360));
    CHECK(centre.depth == doctest::Approx(10));
    CHECK(project(p, cam.intrinsics(), {1, 0, 0}).u > 640);  // +X is to the right when looking north
    CHECK(project(p, cam.intrinsics(), {0, 0, 1}).v < 360);  // up is up

    const Spec att = Spec::fromJson({{"id", "d"}, {"type", "attached"}, {"parent", "car"}, {"offset", {{"pos", {1, 0, 1}}}}});
    Camera dash(att);
    const Pose car{{5, 5, 0}, Quat::axisAngle({0, 0, 1}, kPi / 2)};  // facing +Y
    const Pose dp = dash.update(0, [&](const std::string&) { return car; });
    CHECK(project(dp, dash.intrinsics(), {5, 50, 1}).inFrame(dash.intrinsics()));
    CHECK(project(dp, dash.intrinsics(), {5, -50, 1}).depth < 0);
}
