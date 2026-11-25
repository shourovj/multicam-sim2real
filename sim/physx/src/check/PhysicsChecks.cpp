#include "check/PhysicsChecks.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>

namespace p2s {

namespace {

constexpr double kG = 9.81;
constexpr double kPenetrationM = 0.03;  // legacy audits: 3 cm (PhysX contact offsets are ~1-2 cm)

struct Deepest {
    double depth = 0;
    int frame = -1;
    bool firstInSecond = true;  // the first part's surface is inside the second (else the reverse)
};

double round2(double v) { return std::round(v * 100) / 100; }
double sig3(double v) { return std::isfinite(v) ? std::stod(fmt(v, 3)) : v; }
Json rounded(const Vec3& v) { return {round2(v.x), round2(v.y), round2(v.z)}; }

// Deepest point of part `a`'s surface samples inside part `b` at a frame (0 if none inside).
double depthInside(const Part& a, const Part& b, int f) {
    double worst = 0;
    const PosedShapes bs{b.shapes, b.poses[size_t(f)]};
    for (const Vec3& s : a.samples) worst = std::max(worst, -bs.sdf(a.poses[size_t(f)].transform(s)));
    return worst;
}

bool isKinematic(const EntityInfo& e) {
    return e.kind == "walker" || e.desc.value("type", "") == "kinematic";
}

void checkFinite(const RunData& run, Report& r) {
    for (const Part& p : run.parts())
        for (int f = 0; f < run.frames(); ++f)
            if (!p.poses[size_t(f)].p.finite() || !p.poses[size_t(f)].q.finite()) {
                r.add({"finite", Severity::Fail, p.entity, "pose of " + p.name + " is not finite from t=" + fmt(run.time(f)) + " s", ""});
                return;
            }
    for (const PointSet& ps : run.pointSets())
        for (size_t f = 0; f < ps.frames.size(); ++f)
            for (const Vec3& v : ps.frames[f])
                if (!v.finite()) {
                    r.add({"finite", Severity::Fail, ps.entity, "point positions are not finite from t=" + fmt(run.time(int(f))) + " s", ""});
                    return;
                }
    r.add({"finite", Severity::Pass, "", "all positions finite", ""});
}

void checkSolver(const RunData& run, Report& r) {
    const Json& runInfo = run.x().at("run");
    const int errors = runInfo.value("physx_errors", 0);
    if (errors > 0) {
        std::string msgs;
        for (const Json& m : runInfo.value("physx_messages", Json::array()))
            if (m.get<std::string>().rfind("PhysX error", 0) == 0 && msgs.size() < 600) msgs += "\n    " + m.get<std::string>();
        r.add({"physx_errors", Severity::Fail, "", std::to_string(errors) + " PhysX errors:" + msgs, ""});
    } else {
        r.add({"physx_errors", Severity::Pass, "", "no PhysX errors", ""});
    }
}

// Rigid parts against each other: sweep and prune on x per frame, then signed distances of surface samples.
void checkRigidPenetration(const RunData& run, Report& r) {
    const auto& parts = run.parts();
    std::map<std::pair<size_t, size_t>, Deepest> worst, initial;
    for (int f = 0; f < run.frames(); ++f) {
        std::vector<std::pair<Aabb, size_t>> boxes;
        for (size_t i = 0; i < parts.size(); ++i) {
            const Pose& pose = parts[i].poses[size_t(f)];
            if (pose.p.finite() && pose.q.finite()) boxes.push_back({parts[i].worldBounds(f), i});  // else: "finite" fails
        }
        std::sort(boxes.begin(), boxes.end(), [](const auto& a, const auto& b) { return a.first.lo.x < b.first.lo.x; });
        for (size_t i = 0; i < boxes.size(); ++i)
            for (size_t j = i + 1; j < boxes.size() && boxes[j].first.lo.x <= boxes[i].first.hi.x; ++j) {
                if (!overlaps(boxes[i].first, boxes[j].first)) continue;
                const size_t pa = std::min(boxes[i].second, boxes[j].second), pb = std::max(boxes[i].second, boxes[j].second);
                // Two parts that never move overlap the same way in every frame: frame 0 has measured it.
                if (f > 0 && !parts[pa].moving && !parts[pb].moving) continue;
                if (run.exempt(parts[pa].entity, parts[pb].entity)) continue;
                if (!parts[pa].moving && !parts[pb].moving) continue;  // overlapping scenery (tree crowns) is harmless
                const double ab = depthInside(parts[pa], parts[pb], f), ba = depthInside(parts[pb], parts[pa], f);
                auto& w = f == 0 ? initial[{pa, pb}] : worst[{pa, pb}];
                if (std::max(ab, ba) > w.depth) w = {std::max(ab, ba), f, ab >= ba};
            }
    }
    const auto tolerance = [&](const Part& a, const Part& b) {
        return std::max(0.005, std::min(kPenetrationM, 0.5 * std::min(a.minHalfExtent, b.minHalfExtent)));
    };
    const auto label = [&](const Part& p) { return p.name == "body" ? p.entity : p.entity + "." + p.name; };
    const auto inside = [&](const Part& a, const Part& b, const Deepest& d) {
        const Part &in = d.firstInSecond ? a : b, &out = d.firstInSecond ? b : a;
        if (in.entity == "ground") return label(out) + " reaches " + fmt(d.depth) + " m below the ground surface";
        if (out.entity == "ground") return label(in) + " is " + fmt(d.depth) + " m below the ground surface";
        return label(in) + " is " + fmt(d.depth) + " m inside " + label(out);
    };
    bool any = false;
    std::map<std::pair<size_t, size_t>, double> reported;  // initial overlaps: their depth
    for (const auto& [pair, d] : initial) {
        const Part &a = parts[pair.first], &b = parts[pair.second];
        if (d.depth <= std::min(0.01, tolerance(a, b))) continue;
        any = true;
        reported[pair] = d.depth;
        const std::string apart = fmt(d.depth + 0.005) + " m";
        const std::string fix = a.entity == "ground" ? "raise " + label(b) + " by at least " + apart
                                : b.entity == "ground" ? "raise " + label(a) + " by at least " + apart
                                                       : "move " + label(a) + " or " + label(b) + " apart by at least " + apart;
        r.add({"initial_overlap", Severity::Fail, label(a) + "/" + label(b), inside(a, b, d) + " at t=0", fix,
               {{"depth_m", d.depth}}});
    }
    for (const auto& [pair, d] : worst) {
        const Part &a = parts[pair.first], &b = parts[pair.second];
        const double tol = tolerance(a, b);
        if (d.depth <= tol) continue;
        // A pair that started overlapping is reported once, unless it later went deeper still.
        if (const auto it = reported.find(pair); it != reported.end() && d.depth <= it->second + tol) continue;
        any = true;
        const bool ground = a.entity == "ground" || b.entity == "ground";
        r.add({ground ? "below_ground" : "penetration", Severity::Fail, label(a) + "/" + label(b),
               inside(a, b, d) + " at t=" + fmt(run.time(d.frame)) + " s (tolerance " + fmt(tol) + " m)",
               "see the iteration guide for " + std::string(ground ? "below_ground" : "penetration"),
               {{"depth_m", d.depth}, {"t", run.time(d.frame)}}});
    }
    if (!any) r.add({"penetration", Severity::Pass, "", "no rigid interpenetration beyond tolerance", ""});
}

// Deformable surfaces and particles against rigid parts.
void checkPointPenetration(const RunData& run, Report& r) {
    for (const PointSet& ps : run.pointSets()) {
        const bool particles = ps.kind == "particles";
        if (particles && ps.particleKind == "smoke") continue;
        const double insideDepth = std::max(2 * ps.radius, 0.01);  // a particle deeper than this is inside a solid
        Deepest worst;
        std::string against;
        double worstFraction = 0;
        int fractionFrame = 0;
        const int frameStride = particles ? std::max(1, run.frames() / 24) : 1;
        for (int f = 0; f < run.frames(); f += frameStride) {
            const auto& pts = ps.frames[size_t(f)];
            if (pts.empty()) continue;
            Aabb box;
            for (const Vec3& p : pts) box.add(p);
            std::vector<std::reference_wrapper<const Part>> near;
            for (const Part& part : run.parts())
                if (!run.exempt(ps.entity, part.entity) && overlaps(part.worldBounds(f), box)) near.push_back(part);
            if (near.empty()) continue;
            const size_t stride = std::max<size_t>(1, pts.size() / 2000);
            size_t inside = 0, tested = 0;
            for (size_t i = 0; i < pts.size(); i += stride, ++tested) {
                double deepest = 0;
                size_t by = 0;
                for (size_t k = 0; k < near.size(); ++k)
                    if (const double d = -near[k].get().sdf(f, pts[i]); d > deepest) deepest = d, by = k;
                if (particles) {
                    inside += deepest > insideDepth;
                } else if (deepest > worst.depth) {
                    worst = {deepest, f};
                    against = near[by].get().entity;
                }
            }
            const double frac = double(inside) / double(tested);
            if (particles && frac > worstFraction) worstFraction = frac, fractionFrame = f;
        }
        if (particles) {
            if (worstFraction > 0.02)
                r.add({"particles_in_solids", Severity::Fail, ps.entity,
                       fmt(100 * worstFraction) + "% of " + ps.entity + " particles are inside solids at t=" + fmt(run.time(fractionFrame)) + " s",
                       "thicken walls (>= 2 particle spacings), lower emitter speed, raise substeps"});
        } else if (worst.depth > kPenetrationM) {
            r.add({"penetration", Severity::Fail, ps.entity + "/" + against,
                   ps.entity + " surface is " + fmt(worst.depth) + " m inside " + against + " at t=" + fmt(run.time(worst.frame)) + " s",
                   "add a refine window around the impact, raise substeps or the soft body's youngs_pa / solver_iterations",
                   {{"depth_m", worst.depth}, {"t", run.time(worst.frame)}}});
        }
    }
}

void checkEscaped(const RunData& run, Report& r) {
    for (const Part& p : run.parts()) {
        if (p.entity == "ground") continue;
        for (int f = 0; f < run.frames(); ++f) {
            const Vec3 x = p.poses[size_t(f)].p;
            if (x.norm() > 5000 || (run.hasGround() && x.z < run.groundZ() - 5)) {
                r.add({"escaped", Severity::Fail, p.entity, p.entity + " left the scene at t=" + fmt(run.time(f)) + " s (at " +
                       fmt(x.x) + ", " + fmt(x.y) + ", " + fmt(x.z) + ")", ""});
                break;
            }
        }
    }
    for (const PointSet& ps : run.pointSets()) {
        const auto& last = ps.frames.back();
        if (last.empty() || !run.hasGround()) continue;
        const auto below = std::count_if(last.begin(), last.end(), [&](const Vec3& v) { return v.z < run.groundZ() - 0.5; });
        if (below > long(last.size() / 50))
            r.add({"escaped", Severity::Fail, ps.entity, std::to_string(below) + " of " + std::to_string(last.size()) + " " +
                   ps.entity + " points are below the ground at the end", ""});
    }
}

void checkEnergy(const RunData& run, Report& r) {
    const auto e = run.energy();
    if (e.empty()) return;
    double scale = 1.0;
    for (const auto& row : e) scale = std::max({scale, row[0], std::fabs(row[3])});
    const double e0 = e[0][0] + e[0][1];
    double worst = 0;
    int when = 0;
    for (size_t f = 0; f < e.size(); ++f) {
        if (!std::isfinite(e[f][0] + e[f][1] + e[f][3])) {
            r.add({"finite", Severity::Fail, "energy", "mechanical energy is not finite from t=" + fmt(run.time(int(f))) + " s", ""});
            return;
        }
        // Gained beyond all positive actuator work since t=0 (the work counter may include the pre-roll).
        const double excess = (e[f][0] + e[f][1]) - e0 - (e[f][3] - e[0][3]);
        if (excess > worst) worst = excess, when = int(f);
    }
    const double allowed = 0.03 * scale + 1.0;
    bool kinematicPush = false;  // kinematic bodies do unmetered work on what they push
    for (const Json& c : run.x().at("contacts"))
        for (const char* side : {"a", "b"}) {
            const std::string id = c.at(side).get<std::string>();
            if (run.hasEntity(id) && isKinematic(run.entity(id))) kinematicPush = true;
        }
    r.fact("energy_j", {{"initial", sig3(e0)}, {"final", sig3(e.back()[0] + e.back()[1])},
                        {"actuator_work_in", sig3(e.back()[3] - e[0][3])}, {"max_unexplained_gain", sig3(worst)}});
    if (worst > allowed)
        r.add({"energy", kinematicPush ? Severity::Warn : Severity::Fail, "",
               "mechanical energy rose " + fmt(worst) + " J above the start plus all actuator work (allowed " + fmt(allowed) +
                   " J), peaking at t=" + fmt(run.time(when)) + " s" +
                   (kinematicPush ? "; kinematic bodies pushed others, which is unmetered work" : ""),
               "", {{"excess_j", worst}, {"t", run.time(when)}}});
    else
        r.add({"energy", Severity::Pass, "", "no energy created: max unexplained gain " + fmt(worst) + " J (allowed " + fmt(allowed) + " J)", ""});
}

void checkMotion(const RunData& run, Report& r) {
    Table t{"Motion (judge against the prompt and the real world)", "peak acceleration is |dv/dt| (0 at rest, 1 g in free fall); g = 9.81 m/s^2",
            {"entity", "kind", "mass kg", "peak speed m/s", "peak accel g", "at t s", "in a contact then"}, {}};
    const std::map<std::string, double> typicalSpeed = {{"walker", 3}, {"vehicle", 70}, {"multirotor", 40}, {"tracked_vehicle", 25}};
    for (const auto& [id, e] : run.entities()) {
        if (e.velocity.empty()) continue;
        double vmax = 0, amax = 0;
        int at = 0;
        for (int f = 0; f < run.frames(); ++f) {
            vmax = std::max(vmax, e.velocity[size_t(f)].norm());
            if (f > 0) {
                const double a = ((e.velocity[size_t(f)] - e.velocity[size_t(f - 1)]) * run.fps()).norm();
                if (a > amax) amax = a, at = f;
            }
        }
        amax = std::max(amax, e.desc.value("peak_accel_mps2", 0.0));
        bool contact = false;  // touching anything, the ground included, around the peak
        for (const Json& c : run.x().at("contacts")) {
            const std::string a = c.at("a"), b = c.at("b");
            if ((a == id || b == id) && c.at("first_t").get<double>() <= run.time(at) + 0.2 &&
                c.at("last_t").get<double>() >= run.time(at) - 0.2)
                contact = true;
        }
        t.rows.push_back({id, e.kind, fmt(e.mass), fmt(vmax), fmt(amax / kG), fmt(run.time(at)), contact ? "yes" : "no"});
        if (vmax > 343) {
            r.add({"speed", Severity::Fail, id, id + " exceeded the speed of sound (" + fmt(vmax) + " m/s)", ""});
        } else if (typicalSpeed.count(e.kind) && vmax > typicalSpeed.at(e.kind)) {
            r.add({"speed", Severity::Warn, id, id + " (" + e.kind + ") reached " + fmt(vmax) + " m/s, above the usual " +
                   fmt(typicalSpeed.at(e.kind)) + " m/s", ""});
        }
        if (amax > 2000 * kG) {
            r.add({"acceleration", Severity::Fail, id, id + " peak acceleration " + fmt(amax / kG) + " g: the solver is unstable", ""});
        } else if (amax > 50 * kG && !contact) {
            r.add({"acceleration", Severity::Warn, id, id + " peak acceleration " + fmt(amax / kG) + " g at t=" + fmt(run.time(at)) +
                   " s with no collision then", "", {{"g", amax / kG}}});
        }
    }
    r.table(t);
}

void checkActuators(const RunData& run, Report& r) {
    Table t{"Actuators (judge whether a real machine could do this)", "",
            {"actuator", "kind", "peak", "unit", "per carried mass", "typical", "verdict"}, {}};
    std::map<std::string, std::pair<double, double>> rotorThrust, rotorPower;  // entity -> (sum of peaks, mass)
    const auto verdict = [](double v, double lo, double hi) { return v < lo ? "LOW" : v > hi ? "HIGH" : "ok"; };
    for (const auto& [id, a] : run.x().at("actuators").items()) {
        const std::string kind = a.at("kind"), owner = id.substr(0, id.find('/'));
        const double peak = a.at("peak_force"), power = a.at("peak_power_w"), m = a.at("carried_mass_kg");
        if (kind == "rotor") {
            rotorThrust[owner].first += peak;
            rotorThrust[owner].second += m;
            rotorPower[owner].first += power;
            continue;
        }
        const double r0 = run.hasEntity(owner) ? run.entity(owner).desc.value("wheel_radius_m", 0.0) : 0.0;
        std::string per = "-", typical = "-", v = "-";
        if (kind == "wheel_drive" && m > 0) {
            per = fmt(power / m) + " W/kg";
            typical = "<= 250 W/kg";
            v = verdict(power / m, 0, 250);  // a gently driven vehicle uses little power: only too much is suspicious
            if (r0 > 0) per += ", " + fmt(peak / r0 / m / kG) + " g";
        } else if (kind == "brake" && m > 0 && r0 > 0) {
            per = fmt(peak / r0 / m / kG) + " g";
            typical = "<= 1.2 g";
            v = verdict(peak / r0 / m / kG, 0, 1.2);
        } else if (kind == "traction" && m > 0) {
            per = fmt(peak / m / kG) + " g";
            typical = "<= 0.9 g";
            v = verdict(peak / m / kG, 0, 0.9);
        }
        t.rows.push_back({id, kind, fmt(peak), a.at("force_unit"), per, typical, v});
        if (v == std::string("HIGH") || v == std::string("LOW"))
            r.add({"actuator", Severity::Warn, id, id + " (" + kind + "): " + per + " is outside the typical " + typical, ""});
    }
    // Rotors: judge the craft's capability (thrust-to-weight it was built with), and whether it ran at its limit;
    // using little thrust in a gentle flight is fine.
    for (const auto& [owner, tm] : rotorThrust) {
        const double used = tm.first / (tm.second * kG), sp = rotorPower[owner].first / tm.second;
        const double capable = run.hasEntity(owner) ? run.entity(owner).desc.value("thrust_to_weight", used) : used;
        const bool saturated = used > 0.95 * capable;
        const std::string v = std::string(verdict(capable, 1.3, 5)) + (saturated ? ", at its limit" : "") +
                              (sp > 600 ? ", power HIGH" : "");
        t.rows.push_back({owner + "/rotors", "rotor", fmt(tm.first), "N",
                          "capable T/W " + fmt(capable) + ", used " + fmt(used) + ", " + fmt(sp) + " W/kg",
                          "T/W 1.3-5, <= 600 W/kg", v});
        if (capable < 1.3 || capable > 5 || saturated || sp > 600)
            r.add({"actuator", Severity::Warn, owner + "/rotors",
                   owner + " rotors: built for thrust/weight " + fmt(capable) + ", used up to " + fmt(used) + ", " + fmt(sp) +
                       " W/kg peak (typical T/W 1.3-5; at its limit means the controller asked for more than it has)",
                   ""});
    }
    if (!t.rows.empty()) r.table(t);
}

void checkJitter(const RunData& run, Report& r) {
    const int window = std::min(run.frames() - 1, int(run.fps()));
    if (window < 4) return;
    for (const auto& [id, e] : run.entities()) {
        if (e.velocity.empty() || e.kind == "vehicle" || e.kind == "multirotor") continue;
        const auto p0 = run.position(id, run.frames() - 1 - window), p1 = run.position(id, run.frames() - 1);
        if (!p0 || !p1) continue;
        double mean = 0;
        for (int f = run.frames() - window; f < run.frames(); ++f) mean += e.velocity[size_t(f)].norm() / window;
        if ((*p1 - *p0).norm() < 0.02 && mean > 0.05)
            r.add({"jitter", Severity::Warn, id, id + " stays in place but moves at " + fmt(mean) + " m/s on average over the last second", ""});
    }
}

double tetVolume(const std::vector<Vec3>& v, const std::array<int32_t, 4>& t) {
    return (v[size_t(t[1])] - v[size_t(t[0])]).dot((v[size_t(t[2])] - v[size_t(t[0])]).cross(v[size_t(t[3])] - v[size_t(t[0])])) / 6;
}

void checkSoftBodies(const RunData& run, Report& r) {
    for (const PointSet& ps : run.pointSets()) {
        if (ps.kind != "soft_body" || ps.tets.empty()) continue;
        // Inverted elements: a tetrahedron whose signed volume changed sign.
        std::vector<double> restSign;
        for (const auto& t : ps.tets) restSign.push_back(tetVolume(ps.frames[0], t) >= 0 ? 1 : -1);
        size_t worstFlipped = 0;
        int worstFrame = 0;
        for (int f = 0; f < run.frames(); f += std::max(1, run.frames() / 48)) {
            size_t flipped = 0;
            for (size_t i = 0; i < ps.tets.size(); ++i) flipped += tetVolume(ps.frames[size_t(f)], ps.tets[i]) * restSign[i] < 0;
            if (flipped > worstFlipped) worstFlipped = flipped, worstFrame = f;
        }
        // A few slivers at a contact patch are a local solver error; widespread inversion is a failed solve.
        const double frac = double(worstFlipped) / double(ps.tets.size());
        if (frac > 0.002)
            r.add({"fem_inverted", frac > 0.02 ? Severity::Fail : Severity::Warn, ps.entity,
                   std::to_string(worstFlipped) + " of " + std::to_string(ps.tets.size()) + " tetrahedra of " + ps.entity +
                       " (" + fmt(100 * frac, 2) + "%) are inverted at t=" + fmt(run.time(worstFrame)) + " s (the worst frame)",
                   frac > 0.02 ? "" : "local: acknowledge if the deformation numbers look right, else raise resolution"});
        // Permanent deformation at the end, measured in the host's frame (or the world's).
        const auto hostParts = run.partsOf(ps.host);
        const Pose hp = hostParts.empty() ? Pose{} : hostParts.front().get().poses.back();
        std::vector<double> disp;
        const auto& last = ps.frames.back();
        for (size_t i = 0; i < last.size() && i < ps.rest.size(); ++i)
            disp.push_back(((ps.host.empty() ? last[i] : hp.transformInv(last[i])) - ps.rest[i]).norm());
        if (disp.empty()) continue;
        std::sort(disp.begin(), disp.end());
        const auto pct = [&](double q) { return sig3(disp[size_t(q * double(disp.size() - 1))]); };
        r.fact("deformation_m:" + ps.entity, {{"median", pct(0.5)}, {"p90", pct(0.9)}, {"p99", pct(0.99)}, {"max", pct(1)}});
    }
}

void facts(const RunData& run, Report& r) {
    Json events = Json::array();
    for (const Json& e : run.x().at("events")) events.push_back(e.at("name").get<std::string>() + " @ " + fmt(e.at("t").get<double>()) + " s");
    r.fact("events", events);
    Json bodies = Json::object(), still = Json::array();
    for (const auto& [id, e] : run.entities()) {
        const auto p0 = run.position(id, 0), p1 = run.position(id, run.frames() - 1);
        if (!p0 || id == "ground") continue;
        if (!e.moving) {
            still.push_back(id);
            continue;
        }
        Json b = {{"start", rounded(*p0)}, {"end", p1 ? rounded(*p1) : Json()}};
        if (!e.velocity.empty()) {
            double vmax = 0;
            for (const Vec3& v : e.velocity) vmax = std::max(vmax, v.norm());
            b["peak_speed_mps"] = round2(vmax);
        }
        bodies[id] = b;
    }
    r.fact("bodies", bodies);
    if (!still.empty()) r.fact("did_not_move", still);
    Json contacts = Json::array();
    for (const Json& c : run.x().at("contacts"))
        if (c.at("a") != "ground" && c.at("b") != "ground") {
            const double force = c.at("peak_force_n").get<double>();  // 0: a deformable contact, force not metered
            contacts.push_back({{"pair", c.at("a").get<std::string>() + "+" + c.at("b").get<std::string>()},
                                {"first_t", sig3(c.at("first_t").get<double>())},
                                {"peak_force_n", force > 0 ? Json(sig3(force)) : Json("not measured")}});
        }
    r.fact("contacts", contacts);
}

}  // namespace

void runPhysicsChecks(const RunData& run, Report& r) {
    checkFinite(run, r);
    checkSolver(run, r);
    checkRigidPenetration(run, r);
    checkPointPenetration(run, r);
    checkEscaped(run, r);
    checkEnergy(run, r);
    checkMotion(run, r);
    checkActuators(run, r);
    checkJitter(run, r);
    checkSoftBodies(run, r);
    facts(run, r);
}

}  // namespace p2s
