#include "check/Expectations.h"

#include <map>
#include <set>
#include <sstream>

namespace p2s {

namespace {

struct View {
    double inFrame = 0;  // fraction of the entity's sample points inside the image
    double visible = 0;  // fraction inside the image and not hidden behind another body
    double area = 0;     // screen-space bounding box of the visible points, as a fraction of the image
};

bool rayHitsBox(const Vec3& o, const Vec3& d, double maxT, const Aabb& b) {
    double t0 = 0, t1 = maxT;
    for (int a = 0; a < 3; ++a) {
        const double inv = 1.0 / (std::fabs(d[a]) > 1e-12 ? d[a] : 1e-12);
        double ta = (b.lo[a] - o[a]) * inv, tb = (b.hi[a] - o[a]) * inv;
        if (ta > tb) std::swap(ta, tb);
        t0 = std::max(t0, ta);
        t1 = std::min(t1, tb);
        if (t0 > t1) return false;
    }
    return true;
}

// Visibility of every entity from every camera, computed on demand and cached.
class Visibility {
public:
    explicit Visibility(const RunData& run) : run_(run) {}

    const View& at(const CameraTrack& cam, const std::string& entity, int f) {
        auto& track = cache_[{cam.id, entity}];
        if (track.empty()) track.resize(size_t(run_.frames()));
        auto& slot = track[size_t(f)];
        if (!slot) slot = compute(cam, entity, f);
        return *slot;
    }

private:
    View compute(const CameraTrack& cam, const std::string& entity, int f) const {
        View v;
        const Pose& cp = cam.poses[size_t(f)];
        const auto pts = run_.samples(entity, f, 160);
        if (pts.empty()) return v;
        Aabb screen;
        int in = 0, vis = 0;
        for (const Vec3& p : pts) {
            const Projection pr = project(cp, cam.k, p);
            if (!pr.inFrame(cam.k) || pr.depth > cam.k.farM) continue;
            ++in;
            const Vec3 d = (p - cp.p).normalized();
            const double dist = (p - cp.p).norm();
            bool hidden = false;
            for (const Part& part : run_.parts()) {
                if (part.entity == entity || part.entity == cam.parent || run_.exempt(part.entity, entity)) continue;
                if (!rayHitsBox(cp.p, d, dist, part.worldBounds(f))) continue;
                const auto hit = PosedShapes{part.shapes, part.poses[size_t(f)]}.raycast(cp.p, d, dist);
                if (hit && *hit < dist - 0.05) {
                    hidden = true;
                    break;
                }
            }
            if (hidden) continue;
            ++vis;
            screen.add({pr.u, pr.v, 0});
        }
        v.inFrame = double(in) / double(pts.size());
        v.visible = double(vis) / double(pts.size());
        if (vis > 0) v.area = (screen.extent().x * screen.extent().y) / double(cam.k.width * cam.k.height);
        return v;
    }

    const RunData& run_;
    std::map<std::pair<std::string, std::string>, std::vector<std::optional<View>>> cache_;
};

const CameraTrack& camera(const RunData& run, const Spec& s) {
    const std::string id = s.str("camera");
    for (const CameraTrack& c : run.cameras())
        if (c.id == id) return c;
    s["camera"].fail("no camera '" + id + "' in the scene");
}

// "in 0.00-1.20 s, out 1.20-3.00 s": runs of frames where the predicate holds.
std::string timeline(const RunData& run, const std::vector<bool>& on) {
    std::ostringstream s;
    size_t start = 0;
    for (size_t f = 1; f <= on.size(); ++f)
        if (f == on.size() || on[f] != on[start]) {
            if (start > 0) s << ", ";
            s << (on[start] ? "in " : "out ") << fmt(run.time(int(start)), 3) << "-" << fmt(run.time(int(f - 1)), 3) << " s";
            start = f;
        }
    return s.str();
}

// Where the four corner rays of the image meet the ground: what the camera covers, in words.
std::string footprint(const CameraTrack& cam, const Pose& p, double groundZ) {
    std::ostringstream s;
    const double f = cam.k.focalPx();
    const double corners[4][2] = {{0, 0}, {double(cam.k.width), 0}, {double(cam.k.width), double(cam.k.height)}, {0, double(cam.k.height)}};
    const char* names[4] = {"top-left", "top-right", "bottom-right", "bottom-left"};
    for (int i = 0; i < 4; ++i) {
        const Vec3 dc{(corners[i][0] - cam.k.width / 2.0) / f, -(corners[i][1] - cam.k.height / 2.0) / f, -1};
        const Vec3 d = p.q.rotate(dc).normalized();
        s << names[i] << ": ";
        if (d.z < -1e-6) {
            const double t = (groundZ - p.p.z) / d.z;
            const Vec3 hit = p.p + d * t;
            if (t > cam.k.farM) s << "beyond " << fmt(cam.k.farM) << " m";
            else s << "ground at (" << fmt(hit.x, 3) << ", " << fmt(hit.y, 3) << ")";
        } else {
            s << "sky";
        }
        s << (i < 3 ? "; " : "");
    }
    return s.str();
}

std::pair<int, int> window(const RunData& run, const Spec& s, const std::string& key) {
    if (!s.has(key)) return {0, run.frames() - 1};
    const auto w = s.numbers(key);
    if (w.size() != 2 || w[1] < w[0]) s[key].fail("expected [start_s, end_s]");
    return {run.frameAt(w[0]), run.frameAt(w[1])};
}

std::vector<std::string> referencedEntities(const Spec& scene) {
    std::vector<std::string> out;
    if (!scene.has("expect")) return out;
    for (const Json& e : scene["expect"].opaque())
        for (const char* k : {"entity", "a", "b"})
            if (e.contains(k) && e[k].is_string()) out.push_back(e[k]);
    return out;
}

}  // namespace

void runCameraCoverage(const RunData& run, const Spec& scene, Report& r) {
    Visibility vis(run);
    std::set<std::string> ids;
    for (const auto& id : referencedEntities(scene)) ids.insert(id);
    if (run.entities().size() <= 40)
        for (const auto& [id, e] : run.entities())
            if (id != "ground" && e.kind != "joint" && e.kind != "traffic_signal") ids.insert(id);
    std::ostringstream md;
    const int stride = std::max(1, run.frames() / 48);
    for (const CameraTrack& cam : run.cameras()) {
        Table t{"Camera " + cam.id + ": what it sees", "",
                {"entity", "in view", "visible when in view", "mean screen area", "timeline"}, {}};
        t.note = "Image corners at t=0 (corner rays to the ground): " + footprint(cam, cam.poses[0], run.groundZ()) +
                 ". Camera at (" + fmt(cam.poses[0].p.x, 3) + ", " + fmt(cam.poses[0].p.y, 3) + ", " +
                 fmt(cam.poses[0].p.z, 3) + "), fov " + fmt(cam.k.fovDeg) + " deg, " + std::to_string(cam.k.width) + "x" +
                 std::to_string(cam.k.height) + ".";
        for (const std::string& id : ids) {
            std::vector<bool> on;
            int n = 0, inView = 0;
            double visSum = 0, areaSum = 0;
            for (int f = 0; f < run.frames(); f += stride) {
                const View& v = vis.at(cam, id, f);
                ++n;
                const bool seen = v.visible > 0;
                on.push_back(seen);
                if (seen) {
                    ++inView;
                    visSum += v.visible / std::max(v.inFrame, 1e-9);
                    areaSum += v.area;
                }
            }
            // Timeline on sampled frames, reported in seconds.
            std::vector<bool> full(size_t(run.frames()), false);
            for (int f = 0; f < run.frames(); ++f) full[size_t(f)] = on[size_t(std::min(f / stride, int(on.size()) - 1))];
            t.rows.push_back({id, fmt(100.0 * inView / n) + "%", inView ? fmt(100 * visSum / inView) + "%" : "-",
                              inView ? fmt(100 * areaSum / inView) + "%" : "-", timeline(run, full)});
        }
        r.table(t);
    }
}

void runExpectations(const RunData& run, const Spec& scene, Report& r) {
    Visibility vis(run);
    const std::vector<Spec> expects = scene.items("expect");
    std::set<std::string> camerasCovered;
    bool physical = false;
    int index = 0;
    for (const Spec& s : expects) {
        ++index;
        const std::string type = s.oneOf("type", {"event", "no_event", "event_order", "in_view", "enters_view", "leaves_view",
                                                  "event_in_view", "reaches", "speed", "at_rest", "min_gap", "upright",
                                                  "deformation", "particles_in_region", "obeys_signals"});
        const std::string id = s.str("id", type + std::to_string(index));
        const std::string check = "expect_" + type;
        if (s.has("camera")) camerasCovered.insert(s.str("camera"));
        else physical = true;
        bool ok = false;
        std::string msg;
        if (type == "event" || type == "no_event") {
            const std::string ev = s.str("event");
            const auto t = run.eventTime(ev);
            if (type == "no_event") {
                ok = !t;
                msg = t ? ev + " happened at t=" + fmt(*t) + " s" : ev + " never happened";
            } else {
                const auto w = s.has("within_s") ? s.numbers("within_s") : std::vector<double>{0, 1e9};
                if (w.size() != 2) s["within_s"].fail("expected [start_s, end_s]");
                ok = t && *t >= w[0] && *t <= w[1];
                msg = t ? ev + " at t=" + fmt(*t) + " s (wanted " + fmt(w[0]) + "-" + fmt(w[1]) + " s)" : ev + " never happened";
            }
        } else if (type == "event_order") {
            const auto names = s.strings("events");
            double last = -1;
            ok = true;
            for (const auto& n : names) {
                const auto t = run.eventTime(n);
                msg += (msg.empty() ? "" : ", ") + n + (t ? " @ " + fmt(*t) : " never");
                if (!t || *t < last) ok = false;
                if (t) last = *t;
            }
        } else if (type == "in_view") {
            const CameraTrack& cam = camera(run, s);
            const std::string ent = s.str("entity");
            const auto [f0, f1] = window(run, s, "during_s");
            const double minFrac = s.num("min_fraction", 0.9), minArea = s.num("min_screen_fraction", 0.0);
            const double maxOcc = s.num("max_occluded", 0.5);
            int good = 0, n = 0;
            double area = 0, occ = 0;
            for (int f = f0; f <= f1; ++f, ++n) {
                const View& v = vis.at(cam, ent, f);
                const double occluded = v.inFrame > 0 ? 1 - v.visible / v.inFrame : 1;
                area += v.area;
                occ += occluded;
                if (v.visible > 0 && v.area >= minArea && occluded <= maxOcc) ++good;
            }
            const double frac = n ? double(good) / n : 0;
            ok = frac >= minFrac;
            msg = ent + " well in view of " + cam.id + " in " + fmt(100 * frac) + "% of frames (wanted " + fmt(100 * minFrac) +
                  "%); mean screen area " + fmt(100 * area / std::max(n, 1)) + "%, mean occluded " + fmt(100 * occ / std::max(n, 1)) + "%";
        } else if (type == "enters_view" || type == "leaves_view") {
            const CameraTrack& cam = camera(run, s);
            const std::string ent = s.str("entity");
            // enters: out of view at outside_at_s, in view at some frame up to inside_by_s; leaves: the reverse.
            const bool enters = type == "enters_view";
            const double at = s.num(enters ? "outside_at_s" : "inside_at_s");
            const double by = s.num(enters ? "inside_by_s" : "outside_by_s");
            if (by <= at) s.fail("the second time must be after the first");
            const bool startOk = (vis.at(cam, ent, run.frameAt(at)).visible > 0) != enters;
            bool changed = false;
            for (int f = run.frameAt(at); f <= run.frameAt(by); ++f) changed |= (vis.at(cam, ent, f).visible > 0) == enters;
            ok = startOk && changed;
            msg = ent + (enters ? " enters " : " leaves ") + cam.id + ": " + (enters ? "out of view" : "in view") + " at " + fmt(at) +
                  " s: " + (startOk ? "yes" : "no") + "; " + (enters ? "in view" : "out of view") + " by " + fmt(by) + " s: " +
                  (changed ? "yes" : "no");
        } else if (type == "event_in_view") {
            const CameraTrack& cam = camera(run, s);
            const std::string ev = s.str("event");
            const auto t = run.eventTime(ev);
            if (!t) {
                msg = ev + " never happened";
            } else {
                // A contact happens where its two bodies meet: the closest pair of their sample points.
                const int f = run.frameAt(*t);
                std::vector<std::string> names;
                std::stringstream ss(ev);
                for (std::string part; std::getline(ss, part, ':');) names.push_back(part);
                Vec3 where = cam.poses[size_t(f)].p;
                if (names.size() == 3 && names[0] == "contact") {
                    double best = 1e300;
                    for (const Vec3& a : run.samples(names[1], f, 200))
                        for (const Vec3& b : run.samples(names[2], f, 200))
                            if ((a - b).norm2() < best) best = (a - b).norm2(), where = (a + b) * 0.5;
                } else if (names.size() >= 2) {
                    if (const auto p = run.position(names[0], f)) where = *p;
                }
                const Projection pr = project(cam.poses[size_t(f)], cam.k, where);
                ok = pr.inFrame(cam.k);
                msg = ev + " at t=" + fmt(*t) + " s, at (" + fmt(where.x) + ", " + fmt(where.y) + ", " + fmt(where.z) + "), is " +
                      (ok ? "inside" : "outside") + " " + cam.id + "'s frame";
            }
        } else if (type == "reaches") {
            const std::string ent = s.str("entity");
            const Vec3 target = s.vec3("point");
            const double within = s.positive("within_m"), by = s.num("by_s");
            double best = 1e300;
            for (int f = 0; f <= run.frameAt(by); ++f)
                if (const auto p = run.position(ent, f)) best = std::min(best, (*p - target).norm());
            ok = best <= within;
            msg = ent + " got within " + fmt(best) + " m of the target by " + fmt(by) + " s (wanted " + fmt(within) + " m)";
        } else if (type == "speed" || type == "at_rest") {
            const std::string ent = s.str("entity");
            const auto& vel = run.entity(ent).velocity;
            if (vel.empty()) s["entity"].fail(ent + " has no recorded velocity");
            const auto [f0, f1] = type == "speed" ? window(run, s, "during_s")
                                                  : std::make_pair(run.frameAt(s.num("after_s")), run.frames() - 1);
            const double lo = type == "speed" ? s.num("min_mps", 0) : 0;
            const double hi = type == "speed" ? s.num("max_mps", 1e9) : s.num("max_speed_mps", 0.05);
            double vmin = 1e300, vmax = 0;
            for (int f = f0; f <= f1; ++f) vmin = std::min(vmin, vel[size_t(f)].norm()), vmax = std::max(vmax, vel[size_t(f)].norm());
            ok = vmin >= lo && vmax <= hi;
            msg = ent + " speed " + fmt(vmin) + "-" + fmt(vmax) + " m/s between " + fmt(run.time(f0)) + " and " + fmt(run.time(f1)) +
                  " s (wanted " + fmt(lo) + "-" + fmt(hi) + ")";
        } else if (type == "min_gap") {
            const std::string a = s.str("a"), b = s.str("b");
            const double metres = s.num("metres");
            double best = 1e300;
            int when = 0;
            for (int f = 0; f < run.frames(); f += std::max(1, run.frames() / 96)) {
                const auto pa = run.samples(a, f, 120), pb = run.samples(b, f, 120);
                for (const Vec3& x : pa)
                    for (const Vec3& y : pb)
                        if ((x - y).norm() < best) best = (x - y).norm(), when = f;
            }
            ok = best >= metres;
            msg = a + " and " + b + " came within " + fmt(best) + " m (surface samples) at t=" + fmt(run.time(when)) + " s (wanted >= " + fmt(metres) + ")";
        } else if (type == "upright") {
            const std::string ent = s.str("entity");
            const double maxTilt = s.num("max_tilt_deg", 30);
            double worst = 0;
            for (const Part& p : run.parts())
                if (p.entity == ent) {  // the main part (body, chassis, base link): wheels roll, they do not tip
                    for (const Pose& q : p.poses) worst = std::max(worst, rad2deg(std::acos(std::clamp(q.q.axisZ().z, -1.0, 1.0))));
                    break;
                }
            ok = worst <= maxTilt;
            msg = ent + " tilted up to " + fmt(worst) + " deg (allowed " + fmt(maxTilt) + ")";
        } else if (type == "deformation") {
            const std::string ent = s.str("entity");
            const Json& facts = r.facts();
            const std::string key = "deformation_m:" + ent;
            const double lo = s.num("min_m", 0), hi = s.num("max_m", 1e9);
            if (!facts.contains(key)) s["entity"].fail(ent + " is not a soft body");
            const double p90 = facts.at(key).at("p90");
            ok = p90 >= lo && p90 <= hi;
            msg = ent + " permanent deformation (90th percentile of vertices) " + fmt(p90) + " m (wanted " + fmt(lo) + "-" + fmt(hi) + ")";
        } else if (type == "particles_in_region") {
            const std::string ent = s.str("entity");
            Aabb box;
            box.add(s.vec3("min"));
            box.add(s.vec3("max"));
            const int f = s.has("at_s") ? run.frameAt(s.num("at_s")) : run.frames() - 1;
            const double minFrac = s.num("min_fraction", 0.8);
            size_t inside = 0, total = 0;
            for (const PointSet& ps : run.pointSets())
                if (ps.entity == ent) {
                    total = ps.frames[size_t(f)].size();
                    for (const Vec3& p : ps.frames[size_t(f)]) inside += box.contains(p);
                }
            const double frac = total ? double(inside) / double(total) : 0;
            ok = frac >= minFrac;
            msg = fmt(100 * frac) + "% of " + std::to_string(total) + " " + ent + " particles in the region at t=" + fmt(run.time(f)) +
                  " s (wanted " + fmt(100 * minFrac) + "%)";
        } else {  // obeys_signals
            ok = true;
            for (const Json& e : run.x().at("events")) {
                const std::string n = e.at("name");
                if (n.size() > 17 && n.substr(n.size() - 17) == ":passes_stop_line" && e.value("data", Json::object()).value("light", "") == "red") {
                    ok = false;
                    msg += n + " on red at t=" + fmt(e.at("t").get<double>()) + " s; ";
                }
            }
            if (ok) msg = "no vehicle crossed a stop line on red";
        }
        s.rejectUnknownKeys();
        r.add({check, ok ? Severity::Pass : Severity::Fail, id, msg, ""});
    }
    std::vector<std::string> missing;
    for (const CameraTrack& c : run.cameras())
        if (!camerasCovered.count(c.id)) missing.push_back(c.id);
    if (!missing.empty() || !physical) {
        std::string m = "expect has ";
        if (!physical) m += "no physical outcome";
        for (const auto& c : missing) m += std::string(m.size() > 11 ? ", " : "") + "nothing for camera " + c;
        r.add({"expect_coverage", Severity::Fail, "", m, "add expectations (see the iteration guide)"});
    }
}

}  // namespace p2s
