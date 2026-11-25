// Physics against analytic results, on the GPU. Each test builds a scene, runs it through World into X and reads
// it back with the checker's RunData, so the whole S -> X path is exercised. Tolerances cover the integrator's
// first-order error at the scene's step (dt = 1 / (fps * substeps)).
#include <filesystem>

#include "check/PhysicsChecks.h"
#include "check/RunData.h"
#include "physx/Runtime.h"
#include "third_party/doctest.h"
#include "world/World.h"

using namespace p2s;

namespace {

Runtime& runtime() {
    static Runtime rt;
    return rt;
}

RunData simulate(const std::string& name, Json scene) {
    scene["name"] = name;
    if (!scene.contains("clock")) scene["clock"] = {{"fps", 24}, {"duration_s", 2}, {"substeps", 10}};
    const std::string dir = (std::filesystem::temp_directory_path() / "p2s_tests" / name).string();
    {
        World w(runtime(), Spec::fromJson(scene));
        w.run(dir, Json::object());
    }
    return RunData(dir);
}

Vec3 at(const RunData& r, const std::string& id, double t) { return *r.position(id, r.frameAt(t)); }

Json sphere(const std::string& id, Vec3 pos, double r, double mass, const std::string& material = "default") {
    return {{"id", id}, {"pos", toJson(pos)}, {"mass_kg", mass},
            {"shapes", {{{"type", "sphere"}, {"radius", r}, {"material", material}}}}};
}

}  // namespace

TEST_CASE("free fall matches h - g t^2 / 2") {
    const auto r = simulate("free_fall", {{"world", {{"ground", false}}}, {"bodies", {sphere("ball", {0, 0, 10}, 0.1, 1)}}});
    const double dt = 1.0 / 240;
    CHECK(std::fabs(at(r, "ball", 1.0).z - (10 - 0.5 * 9.81)) < 0.5 * 9.81 * dt * 1.0 + 1e-3);
}

TEST_CASE("projectile follows a parabola") {
    Json ball = sphere("ball", {0, 0, 0.5}, 0.05, 1);
    ball["velocity"] = {10, 0, 10};
    const auto r = simulate("projectile", {{"world", {{"ground", false}}}, {"bodies", {ball}}});
    const Vec3 p = at(r, "ball", 2.0);
    CHECK(p.x == doctest::Approx(20).epsilon(0.005));
    CHECK(std::fabs(p.z - (0.5 + 20 - 0.5 * 9.81 * 4)) < 0.05);
}

TEST_CASE("a bouncing ball returns to about e^2 of its drop height") {
    const Json mats = {{{"id", "bouncy"}, {"preset", "rubber"}, {"restitution", 0.8}, {"restitution_combine", "min"}}};
    const auto r = simulate("bounce", {{"clock", {{"fps", 60}, {"duration_s", 2}, {"substeps", 8}}},
                                       {"materials", mats},
                                       {"world", {{"ground", {{"material", "bouncy"}}}}},
                                       {"bodies", {sphere("ball", {0, 0, 1.1}, 0.1, 1, "bouncy")}}});
    double peak = 0;  // highest point after the first bounce (ball falls 1 m to touch down at ~0.45 s)
    for (int f = r.frameAt(0.6); f < r.frameAt(1.6); ++f) peak = std::max(peak, r.position("ball", f)->z - 0.1);
    CHECK(peak == doctest::Approx(0.64).epsilon(0.15));
}

TEST_CASE("a pendulum swings with period 2 pi sqrt(L / g)") {
    const double L = 1.0, a = deg2rad(8);
    const auto r = simulate("pendulum", {{"clock", {{"fps", 100}, {"duration_s", 6}, {"substeps", 5}}},
                                         {"world", {{"ground", false}}},
                                         {"bodies", {sphere("bob", {L * std::sin(a), 0, 5 - L * std::cos(a)}, 0.02, 1)}},
                                         {"joints", {{{"id", "pivot"}, {"type", "spherical"}, {"a", "world"}, {"b", "bob"}, {"anchor", {0, 0, 5}}}}}});
    std::vector<double> crossings;  // downward zero crossings of x
    for (int f = 1; f < r.frames(); ++f)
        if (r.position("bob", f - 1)->x > 0 && r.position("bob", f)->x <= 0) crossings.push_back(r.time(f));
    REQUIRE(crossings.size() >= 2);
    const double period = (crossings.back() - crossings.front()) / double(crossings.size() - 1);
    CHECK(period == doctest::Approx(2 * kPi * std::sqrt(L / 9.81) * 1.0012).epsilon(0.01));
}

TEST_CASE("a mass on a spring oscillates with period 2 pi sqrt(m / k)") {
    const double k = 100, m = 1;
    const auto r = simulate("spring", {{"clock", {{"fps", 200}, {"duration_s", 3}, {"substeps", 5}}},
                                       {"world", {{"ground", false}, {"gravity", {0, 0, 0}}}},
                                       {"bodies", {sphere("mass", {1.2, 0, 0}, 0.02, m)}},
                                       {"joints", {{{"id", "spring"}, {"type", "distance"}, {"a", "world"}, {"b", "mass"},
                                                    {"anchor", {0, 0, 0}}, {"min_distance", 1.0}, {"max_distance", 1.0},
                                                    {"spring", {{"stiffness", k}, {"damping", 0}}}}}}});
    std::vector<double> peaks;
    for (int f = 1; f + 1 < r.frames(); ++f) {
        const double x0 = r.position("mass", f - 1)->x, x1 = r.position("mass", f)->x, x2 = r.position("mass", f + 1)->x;
        if (x1 > x0 && x1 >= x2 && x1 > 1.1) peaks.push_back(r.time(f));
    }
    REQUIRE(peaks.size() >= 2);
    const double period = (peaks.back() - peaks.front()) / double(peaks.size() - 1);
    CHECK(period == doctest::Approx(2 * kPi * std::sqrt(m / k)).epsilon(0.03));
}

TEST_CASE("a sliding box stops after v^2 / (2 mu g)") {
    const Json mats = {{{"id", "slide"}, {"static_friction", 0.5}, {"dynamic_friction", 0.5}, {"restitution", 0}}};
    Json box = {{"id", "box"}, {"pos", {0, 0, 0.25}}, {"mass_kg", 10}, {"velocity", {5, 0, 0}},
                {"shapes", {{{"type", "box"}, {"size", {0.5, 0.5, 0.5}}, {"material", "slide"}}}}};
    const auto r = simulate("friction", {{"clock", {{"fps", 50}, {"duration_s", 2}, {"substeps", 10}}},
                                         {"materials", mats}, {"world", {{"ground", {{"material", "slide"}}}}}, {"bodies", {box}}});
    CHECK(at(r, "box", 2.0).x == doctest::Approx(25.0 / (2 * 0.5 * 9.81)).epsilon(0.08));
}

TEST_CASE("a free spinning body keeps its kinetic energy") {
    Json box = {{"id", "brick"}, {"pos", {0, 0, 0}}, {"mass_kg", 2}, {"angular_velocity", {0.01, 5, 0.01}},
                {"shapes", {{{"type", "box"}, {"size", {0.1, 0.2, 0.3}}}}}};
    const auto r = simulate("spin", {{"clock", {{"fps", 50}, {"duration_s", 5}, {"substeps", 20}}},
                                     {"world", {{"ground", false}, {"gravity", {0, 0, 0}}}}, {"bodies", {box}}});
    const auto e = r.energy();
    CHECK(e.back()[0] == doctest::Approx(e.front()[0]).epsilon(0.03));
}

TEST_CASE("a hovering drone's rotors carry exactly its weight") {
    Json drone = {{"id", "drone"}, {"pos", {0, 0, 2}}, {"mass_kg", 1.5},
                  {"shapes", {{{"type", "box"}, {"size", {0.3, 0.3, 0.08}}}}},
                  {"rotors", {{"layout", "quad_x"}, {"arm_m", 0.25}}},
                  {"controller", {{"goals", {{{"from_t", 0}, {"t", 1}, {"pos", {0, 0, 2}}}}}}}};
    const auto r = simulate("hover", {{"clock", {{"fps", 24}, {"duration_s", 3}, {"substeps", 20}}}, {"multirotors", {drone}}});
    CHECK((at(r, "drone", 3.0) - Vec3{0, 0, 2}).norm() < 0.05);
    double peak = 0;
    for (const auto& [id, a] : r.x().at("actuators").items()) peak += a.at("peak_force").get<double>();
    CHECK(peak >= 1.5 * 9.81 * 0.98);
}

TEST_CASE("a car cruises at its set speed along its lane") {
    Json car = {{"id", "car"}, {"pos", {0, 0, 0}}, {"speed_mps", 12}, {"mass_kg", 1500},
                {"shapes", {{{"type", "box"}, {"size", {4.4, 1.8, 0.7}}, {"pos", {0, 0, 0.75}}}}},
                {"wheels", {{"radius_m", 0.33}, {"axles", {{{"x", 1.35}, {"track_m", 1.55}, {"steer", true}},
                                                            {{"x", -1.35}, {"track_m", 1.55}}}}}},
                {"limits", {{"max_wheel_torque_nm", 1200}, {"max_brake_torque_nm", 3000}}},
                {"driver", {{"path", {{0, 0}, {200, 0}}}, {"cruise_mps", 12}}}};
    const auto r = simulate("cruise", {{"clock", {{"fps", 24}, {"duration_s", 4}, {"substeps", 20}}}, {"vehicles", {car}}});
    const auto& v = r.entity("car").velocity;
    CHECK(v.back().norm() == doctest::Approx(12).epsilon(0.05));
    CHECK(std::fabs(at(r, "car", 4.0).y) < 0.3);
}

TEST_CASE("a joint loaded beyond its break force snaps") {
    const Json rope = {{"id", "rope"}, {"type", "fixed"}, {"a", "world"}, {"b", "weight"}, {"anchor", {0, 0, 4.5}},
                       {"break_force", 50}};
    Json scene = {{"world", {{"ground", false}}}};
    scene["bodies"] = Json::array({sphere("weight", {0, 0, 4}, 0.1, 10)});
    scene["joints"] = Json::array({rope});
    const auto r = simulate("break", scene);
    CHECK(r.eventTime("rope:broken").has_value());
    CHECK(at(r, "weight", 2.0).z < 0);
}

TEST_CASE("plastic FEM stays crushed, elastic FEM springs back") {
    auto cube = [](bool plastic) {
        Json c = {{"id", "cube"}, {"shape", {{"type", "box"}, {"size", {1, 1, 1}}}}, {"pos", {0, 0, 0}}, {"velocity", {10, 0, 0}},
                  {"density", 500}, {"youngs_pa", 1e7}, {"resolution", 12}, {"solver_iterations", 20}, {"damping", 0.002}};
        if (plastic) c["plastic"] = {{"yield_strain", 0.002}};
        return c;
    };
    const Json wall = {{"id", "wall"}, {"type", "static"}, {"pos", {2, 0, 0}}, {"shapes", {{{"type", "box"}, {"half_extents", {0.5, 2, 2}}}}}};
    const Json base = {{"clock", {{"fps", 24}, {"duration_s", 1}, {"substeps", 10}}},
                       {"world", {{"ground", false}, {"gravity", {0, 0, 0}}}}, {"bodies", {wall}}};
    Json e = base, p = base;
    e["soft_bodies"] = {cube(false)};
    p["soft_bodies"] = {cube(true)};
    const auto length = [](const RunData& r) {
        const auto& pts = r.pointSets().front().frames.back();
        double lo = 1e9, hi = -1e9;
        for (const Vec3& v : pts) lo = std::min(lo, v.x), hi = std::max(hi, v.x);
        return hi - lo;
    };
    CHECK(length(simulate("fem_elastic", e)) > 0.99);
    CHECK(length(simulate("fem_plastic", p)) < 0.97);
}

TEST_CASE("water at rest in a glass stays in it") {
    const Json glass = {{"id", "glass"}, {"type", "static"}, {"pos", {0, 0, 0}},
                        {"shapes", {{{"type", "container"}, {"radius", 0.15}, {"height", 0.25}, {"wall", 0.02}}}}};
    const Json water = {{"id", "water"}, {"preset", "water"}, {"spacing_m", 0.01},
                        {"blocks", {{{"min", {-0.08, -0.08, 0.03}}, {"max", {0.08, 0.08, 0.12}}}}}};
    const auto r = simulate("glass", {{"bodies", {glass}}, {"particles", {water}}});
    const auto& last = r.pointSets().front().frames.back();
    size_t inside = 0;
    for (const Vec3& v : last) inside += std::hypot(v.x, v.y) < 0.15 && v.z > 0;
    CHECK(double(inside) / double(last.size()) > 0.98);
}

TEST_CASE("the checker reports bodies that start overlapping") {
    Json a = {{"id", "a"}, {"pos", {0, 0, 1}}, {"mass_kg", 1}, {"shapes", {{{"type", "box"}, {"size", {1, 1, 1}}}}}};
    Json b = a;
    b["id"] = "b";
    b["pos"] = {0.3, 0, 1};
    const auto r = simulate("overlap", {{"world", {{"ground", false}, {"gravity", {0, 0, 0}}}}, {"bodies", {a, b}}});
    Report report;
    runPhysicsChecks(r, report);
    const Json j = report.toJson(1);
    bool found = false;
    for (const Json& f : j.at("findings")) found |= f.at("check") == "initial_overlap";
    CHECK(found);
}
