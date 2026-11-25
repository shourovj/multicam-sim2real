#include "check/Expectations.h"

#include "check/Visibility.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <sstream>

namespace p2s {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr size_t kMaxListedMoving = 40;  // coverage rows for moving entities that no expectation names

std::string percent(double fraction) { return fraction > 0 && fraction < 1e-4 ? "<0.01%" : fmt(100 * fraction) + "%"; }

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    for (std::string part; std::getline(ss, part, sep);) out.push_back(part);
    return out;
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
    if (p.p.z <= groundZ) return "the camera is below the ground (z " + fmt(p.p.z) + " m)";
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

std::string where(const Vec3& p) { return "(" + fmt(p.x, 3) + ", " + fmt(p.y, 3) + ", " + fmt(p.z, 3) + ")"; }

std::pair<int, int> window(const RunData& run, const Spec& s, const std::string& key) {
    if (!s.has(key)) return {0, run.frames() - 1};
    const auto w = s.numbers(key);
    if (w.size() != 2 || w[1] < w[0]) s[key].fail("expected [start_s, end_s]");
    return {run.frameAt(w[0]), run.frameAt(w[1])};
}

const CameraTrack& camera(const RunData& run, const Spec& s) {
    const std::string id = s.str("camera");
    for (const CameraTrack& c : run.cameras())
        if (c.id == id) return c;
    s["camera"].fail("no camera '" + id + "' in the scene");
}

std::string entityId(const RunData& run, const Spec& s, const std::string& key) {
    const std::string id = s.str(key);
    if (!run.hasEntity(id)) s[key].fail("no entity '" + id + "' in the scene");
    return id;
}

// A contact event with an unknown id or ids out of order would silently never happen.
void checkEventName(const RunData& run, const Spec& at, const std::string& name) {
    if (name.rfind("contact:", 0) != 0) return;
    const auto ids = split(name, ':');
    if (ids.size() != 3) at.fail("contact events are named contact:<a>:<b>, not '" + name + "'");
    for (size_t i = 1; i < 3; ++i)
        if (!run.hasEntity(ids[i].substr(0, ids[i].find('.'))))
            at.fail("'" + name + "' names no entity '" + ids[i] + "'");
    if (ids[1] > ids[2]) at.fail("contact events sort their ids alphabetically: contact:" + ids[2] + ":" + ids[1]);
}

std::string eventName(const RunData& run, const Spec& s) {
    const std::string name = s.str("event");
    checkEventName(run, s["event"], name);
    return name;
}

// Where an event happens: the closest pair of a contact's surface samples, else the position of the first entity
// its name mentions ("truck:stops", "trigger:truck:release").
std::optional<Vec3> eventPlace(const RunData& run, const std::string& name, int f) {
    const auto ids = split(name, ':');
    if (ids.size() == 3 && ids[0] == "contact") {
        double best = kInf;
        Vec3 place;
        for (const Vec3& a : run.samples(ids[1], f, 200))
            for (const Vec3& b : run.samples(ids[2], f, 200))
                if ((a - b).norm2() < best) best = (a - b).norm2(), place = (a + b) * 0.5;
        if (best < kInf) return place;
    }
    for (const std::string& id : ids)
        if (const auto p = run.position(id, f)) return p;
    return std::nullopt;
}

double boxGap(const Aabb& a, const Aabb& b) {
    const Vec3 gap = (b.lo - a.hi).cwiseMax(a.lo - b.hi).cwiseMax({0, 0, 0});
    return gap.norm();
}

// Entities named by expectations (entity, a, b, and the ids in contact events).
std::set<std::string> referencedEntities(const Spec& scene) {
    std::set<std::string> out;
    if (!scene.has("expect")) return out;
    const auto addEvent = [&](const Json& name) {
        if (!name.is_string()) return;
        const auto ids = split(name.get<std::string>(), ':');
        if (ids.size() == 3 && ids[0] == "contact") out.insert({ids[1], ids[2]});
    };
    for (const Json& e : scene["expect"].opaque()) {
        if (!e.is_object()) continue;
        for (const char* k : {"entity", "a", "b"})
            if (e.contains(k) && e[k].is_string()) out.insert(e[k].get<std::string>());
        if (e.contains("event")) addEvent(e["event"]);
        if (e.contains("events") && e["events"].is_array())
            for (const Json& n : e["events"]) addEvent(n);
    }
    return out;
}

struct Outcome {
    bool ok = false;
    std::string message;
};

// Evaluates one expectation of each type against X.
class Judge {
public:
    Judge(const RunData& run, const Report& report) : run_(run), report_(report), vis_(run) {}

    Outcome operator()(const std::string& type, const Spec& s) {
        if (type == "event") return event(s);
        if (type == "no_event") return noEvent(s);
        if (type == "event_order") return eventOrder(s);
        if (type == "in_view") return inView(s);
        if (type == "enters_view" || type == "leaves_view") return entersOrLeaves(s, type == "enters_view");
        if (type == "event_in_view") return eventInView(s);
        if (type == "reaches") return reaches(s);
        if (type == "speed" || type == "at_rest") return speed(s, type == "at_rest");
        if (type == "min_gap") return minGap(s);
        if (type == "upright") return upright(s);
        if (type == "spin") return spin(s);
        if (type == "turns") return turns(s);
        if (type == "deformation") return deformation(s);
        if (type == "particles_in_region") return particlesInRegion(s);
        return obeysSignals(s);
    }

private:
    bool seen(const CameraTrack& cam, const std::string& id, int f) { return vis_.at(cam, id, f).visible > 0; }

    // Bounds of an entity at a frame: its rigid parts and its sample points.
    Aabb bounds(const std::string& id, const std::vector<Vec3>& samples, int f) const {
        Aabb box;
        for (const Vec3& p : samples) box.add(p);
        for (const Part& p : run_.partsOf(id)) {
            const Aabb b = p.worldBounds(f);
            box.add(b.lo);
            box.add(b.hi);
        }
        return box;
    }

    // Surface distance between two entities (0 if touching or overlapping): one's sample points against the
    // other's rigid shapes, both ways; point to point when neither is rigid.
    double gap(const std::string& a, const std::vector<Vec3>& pa, const std::string& b, const std::vector<Vec3>& pb,
               int f) const {
        double best = kInf;
        for (const Part& p : run_.partsOf(b))
            for (const Vec3& x : pa) best = std::min(best, p.sdf(f, x));
        for (const Part& p : run_.partsOf(a))
            for (const Vec3& y : pb) best = std::min(best, p.sdf(f, y));
        if (best == kInf)
            for (const Vec3& x : pa)
                for (const Vec3& y : pb) best = std::min(best, (x - y).norm());
        return std::max(best, 0.0);
    }

    Outcome event(const Spec& s) {
        const std::string ev = eventName(run_, s);
        const auto t = run_.eventTime(ev);
        const auto w = s.has("within_s") ? s.numbers("within_s") : std::vector<double>{};
        if (!t) return {false, ev + " never happened"};
        if (w.empty()) return {true, ev + " at t=" + fmt(*t) + " s"};
        if (w.size() != 2 || w[1] < w[0]) s["within_s"].fail("expected [start_s, end_s]");
        return {*t >= w[0] && *t <= w[1], ev + " at t=" + fmt(*t) + " s (wanted " + fmt(w[0]) + "-" + fmt(w[1]) + " s)"};
    }

    Outcome noEvent(const Spec& s) {
        const std::string ev = eventName(run_, s);
        const auto t = run_.eventTime(ev);
        return {!t, t ? ev + " happened at t=" + fmt(*t) + " s" : ev + " never happened"};
    }

    Outcome eventOrder(const Spec& s) {
        const auto names = s.strings("events");
        Outcome out{true, ""};
        double last = -kInf;
        for (size_t i = 0; i < names.size(); ++i) {
            checkEventName(run_, s["events"].at(i), names[i]);
            const auto t = run_.eventTime(names[i]);
            out.message += (out.message.empty() ? "" : ", ") + names[i] + (t ? " @ " + fmt(*t) : " never");
            if (!t || *t < last) out.ok = false;
            if (t) last = *t;
        }
        return out;
    }

    Outcome inView(const Spec& s) {
        const CameraTrack& cam = camera(run_, s);
        const std::string ent = entityId(run_, s, "entity");
        const auto [f0, f1] = window(run_, s, "during_s");
        const double minFrac = s.num("min_fraction", 0.9), minArea = s.num("min_screen_fraction", 0.0);
        const double maxOcc = s.num("max_occluded", 0.5);
        int good = 0, n = 0;
        double area = 0, occ = 0;
        for (int f = f0; f <= f1; ++f, ++n) {
            const View& v = vis_.at(cam, ent, f);
            const double occluded = v.inFrame > 0 ? 1 - v.visible / v.inFrame : 1;
            area += v.area;
            occ += occluded;
            if (v.visible > 0 && v.area >= minArea && occluded <= maxOcc) ++good;
        }
        const double frac = double(good) / n;
        return {frac >= minFrac, ent + " well in view of " + cam.id + " in " + percent(frac) + " of frames " +
                                     fmt(run_.time(f0)) + "-" + fmt(run_.time(f1)) + " s (wanted " + percent(minFrac) +
                                     "); mean screen area " + percent(area / n) + ", mean occluded " + percent(occ / n)};
    }

    // enters: out of view at the first time, in view at some frame up to the second; leaves: the reverse.
    Outcome entersOrLeaves(const Spec& s, bool enters) {
        const CameraTrack& cam = camera(run_, s);
        const std::string ent = entityId(run_, s, "entity");
        const double at = s.num(enters ? "outside_at_s" : "inside_at_s");
        const double by = s.num(enters ? "inside_by_s" : "outside_by_s");
        if (by <= at) s.fail("the second time must be after the first");
        const bool startOk = seen(cam, ent, run_.frameAt(at)) != enters;
        bool changed = false;
        for (int f = run_.frameAt(at); f <= run_.frameAt(by) && !changed; ++f) changed = seen(cam, ent, f) == enters;
        const std::string before = enters ? "out of view" : "in view", after = enters ? "in view" : "out of view";
        return {startOk && changed, ent + (enters ? " enters " : " leaves ") + cam.id + ": " + before + " at " + fmt(at) +
                                        " s: " + (startOk ? "yes" : "no") + "; " + after + " by " + fmt(by) + " s: " +
                                        (changed ? "yes" : "no")};
    }

    Outcome eventInView(const Spec& s) {
        const CameraTrack& cam = camera(run_, s);
        const std::string ev = eventName(run_, s);
        const auto t = run_.eventTime(ev);
        if (!t) return {false, ev + " never happened"};
        const int f = run_.frameAt(*t);
        const auto place = eventPlace(run_, ev, f);
        if (!place) s["event"].fail("cannot tell where '" + ev + "' happens: use a contact or <entity>:<what> event");
        const bool ok = inClip(project(cam.poses[size_t(f)], cam.k, *place), cam.k);
        return {ok, ev + " at t=" + fmt(*t) + " s, at " + where(*place) + ", is " + (ok ? "inside " : "outside ") +
                        cam.id + "'s frame"};
    }

    Outcome reaches(const Spec& s) {
        const std::string ent = entityId(run_, s, "entity");
        const Vec3 target = s.vec3("point");
        const double within = s.positive("within_m"), by = s.num("by_s");
        double best = kInf;
        for (int f = 0; f <= run_.frameAt(by); ++f)
            if (const auto p = run_.position(ent, f)) best = std::min(best, (*p - target).norm());
        if (best == kInf) s["entity"].fail(ent + " has no position (no body or points)");
        return {best <= within, ent + " got within " + fmt(best) + " m of " + where(target) + " by " + fmt(by) +
                                    " s (wanted " + fmt(within) + " m)"};
    }

    Outcome speed(const Spec& s, bool atRest) {
        const std::string ent = entityId(run_, s, "entity");
        const auto& vel = run_.entity(ent).velocity;
        if (vel.empty()) s["entity"].fail(ent + " has no recorded velocity");
        const auto [f0, f1] = atRest ? std::make_pair(run_.frameAt(s.num("after_s")), run_.frames() - 1)
                                     : window(run_, s, "during_s");
        const double lo = atRest ? 0 : s.num("min_mps", 0);
        const double hi = atRest ? s.num("max_speed_mps", 0.05) : s.num("max_mps", 1e9);
        double vmin = kInf, vmax = 0;
        for (int f = f0; f <= f1; ++f) {
            vmin = std::min(vmin, vel[size_t(f)].norm());
            vmax = std::max(vmax, vel[size_t(f)].norm());
        }
        return {vmin >= lo && vmax <= hi, ent + " speed " + fmt(vmin) + "-" + fmt(vmax) + " m/s between " +
                                              fmt(run_.time(f0)) + " and " + fmt(run_.time(f1)) + " s (wanted " + fmt(lo) +
                                              "-" + fmt(hi) + " m/s)"};
    }

    Outcome minGap(const Spec& s) {
        const std::string a = entityId(run_, s, "a"), b = entityId(run_, s, "b");
        const double metres = s.num("metres");
        double best = kInf;
        int when = 0;
        for (int f = 0; f < run_.frames(); ++f) {
            const auto pa = run_.samples(a, f, 120), pb = run_.samples(b, f, 120);
            if (pa.empty() || pb.empty() || boxGap(bounds(a, pa, f), bounds(b, pb, f)) >= best) continue;
            if (const double d = gap(a, pa, b, pb, f); d < best) best = d, when = f;
        }
        if (best == kInf) s.fail(a + " or " + b + " has no geometry to measure a gap from");
        // The simulator's contact (which includes parts mounted on them, like a crumple zone) means they touched.
        if (const auto t = run_.eventTime("contact:" + std::min(a, b) + ":" + std::max(a, b)))
            best = 0, when = run_.frameAt(*t);
        return {best >= metres, a + " and " + b + " came within " + fmt(best) + " m at t=" + fmt(run_.time(when)) +
                                    " s (wanted >= " + fmt(metres) + " m)"};
    }

    // Rotation rate of the main part from frame to frame (rpm); above half a turn per frame it aliases.
    Outcome spin(const Spec& s) {
        const std::string ent = entityId(run_, s, "entity");
        const auto parts = run_.partsOf(ent);
        if (parts.empty()) s["entity"].fail(ent + " has no rigid body to measure spin on");
        const auto [f0, f1] = window(run_, s, "during_s");
        const double lo = s.num("min_rpm", 0), hi = s.num("max_rpm", 1e9);
        const auto& poses = parts.front().get().poses;
        if (f1 < std::max(f0, 1) || run_.frames() < 2) s["during_s"].fail("needs at least two frames");
        const double toRpm = 60.0 / (2 * kPi) / (run_.time(1) - run_.time(0));
        double rmin = kInf, rmax = 0;
        for (int f = std::max(f0, 1); f <= f1; ++f) {
            const Quat d = (poses[size_t(f - 1)].q.conj() * poses[size_t(f)].q).normalized();
            const double rpm = 2 * std::acos(std::clamp(std::fabs(d.w), 0.0, 1.0)) * toRpm;
            rmin = std::min(rmin, rpm);
            rmax = std::max(rmax, rpm);
        }
        const double aliasRpm = kPi * toRpm;  // half a turn per frame
        std::string msg = ent + " spins at " + fmt(rmin) + "-" + fmt(rmax) + " rpm between " + fmt(run_.time(std::max(f0, 1))) +
                          " and " + fmt(run_.time(f1)) + " s (wanted " + fmt(lo) + "-" + fmt(hi) + " rpm)";
        if (rmax > 0.8 * aliasRpm)
            msg += "; near or above " + fmt(aliasRpm) + " rpm (half a turn per frame) the video cannot show the rotation";
        return {rmin >= lo && rmax <= hi, msg};
    }

    // Largest heading change of the main part about the vertical from the window's start (a car spun round by
    // a crash, a door swinging open), following the rotation through full turns.
    Outcome turns(const Spec& s) {
        const std::string ent = entityId(run_, s, "entity");
        const auto parts = run_.partsOf(ent);
        if (parts.empty()) s["entity"].fail(ent + " has no rigid body to measure its heading on");
        const auto [f0, f1] = window(run_, s, "during_s");
        const double lo = s.num("min_deg", 0), hi = s.num("max_deg", 1e9);
        const auto& poses = parts.front().get().poses;
        const auto heading = [&](int f) {
            const Vec3 x = poses[size_t(f)].q.rotate({1, 0, 0});
            return std::atan2(x.y, x.x);
        };
        double total = 0, most = 0;
        for (int f = f0 + 1; f <= f1; ++f) {
            total += std::remainder(heading(f) - heading(f - 1), 2 * kPi);
            most = std::max(most, std::fabs(total));
        }
        const double deg = rad2deg(most);
        return {deg >= lo && deg <= hi, ent + " turned " + fmt(deg) + " deg about the vertical between " + fmt(run_.time(f0)) +
                                            " and " + fmt(run_.time(f1)) + " s (wanted " + fmt(lo) + "-" + fmt(hi) + " deg)"};
    }

    Outcome upright(const Spec& s) {
        const std::string ent = entityId(run_, s, "entity");
        const double maxTilt = s.num("max_tilt_deg", 30);
        const auto parts = run_.partsOf(ent);
        if (parts.empty()) s["entity"].fail(ent + " has no rigid body to measure tilt on");
        double worst = 0;
        int when = 0;
        const auto& poses = parts.front().get().poses;  // the main part (body, chassis, base link): wheels roll
        for (size_t f = 0; f < poses.size(); ++f)
            if (const double tilt = rad2deg(std::acos(std::clamp(poses[f].q.axisZ().z, -1.0, 1.0))); tilt > worst)
                worst = tilt, when = int(f);
        return {worst <= maxTilt, ent + " tilted up to " + fmt(worst) + " deg at t=" + fmt(run_.time(when)) +
                                      " s (allowed " + fmt(maxTilt) + " deg)"};
    }

    Outcome deformation(const Spec& s) {
        const std::string ent = entityId(run_, s, "entity");
        const std::string key = "deformation_m:" + ent;
        const double lo = s.num("min_m", 0), hi = s.num("max_m", 1e9);
        if (!report_.facts().contains(key)) s["entity"].fail(ent + " is not a soft body with rest positions");
        const double p90 = report_.facts().at(key).at("p90");
        return {p90 >= lo && p90 <= hi, ent + " permanent deformation (90th percentile of vertices) " + fmt(p90) +
                                            " m (wanted " + fmt(lo) + "-" + fmt(hi) + " m)"};
    }

    Outcome particlesInRegion(const Spec& s) {
        const std::string ent = entityId(run_, s, "entity");
        Aabb box;
        box.add(s.vec3("min"));
        box.add(s.vec3("max"));
        const int f = s.has("at_s") ? run_.frameAt(s.num("at_s")) : run_.frames() - 1;
        const double minFrac = s.num("min_fraction", 0.8);
        const auto& sets = run_.pointSets();
        const auto it = std::find_if(sets.begin(), sets.end(), [&](const PointSet& p) { return p.entity == ent; });
        if (it == sets.end()) s["entity"].fail(ent + " is not particles, cloth or a soft body");
        const auto& pts = it->frames[size_t(f)];
        const auto inside = std::count_if(pts.begin(), pts.end(), [&](const Vec3& p) { return box.contains(p); });
        const double frac = pts.empty() ? 0 : double(inside) / double(pts.size());
        return {frac >= minFrac, percent(frac) + " of " + std::to_string(pts.size()) + " " + ent +
                                     " points in the region at t=" + fmt(run_.time(f)) + " s (wanted " + percent(minFrac) + ")"};
    }

    // Every vehicle, or only those listed (a scene where one car runs the red checks the others).
    Outcome obeysSignals(const Spec& s) {
        Outcome out{true, ""};
        std::set<std::string> only;
        if (s.has("vehicles"))
            for (const std::string& v : s.strings("vehicles")) {
                if (!run_.hasEntity(v)) s["vehicles"].fail("no entity '" + v + "'");
                only.insert(v);
            }
        const std::string suffix = ":passes_stop_line";
        for (const Json& e : run_.x().at("events")) {
            const std::string n = e.at("name");
            const std::string who = n.substr(0, n.find(':'));
            if (n.size() > suffix.size() && n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0 &&
                (only.empty() || only.count(who)) && e.value("data", Json::object()).value("light", "") == "red") {
                out.ok = false;
                out.message += (out.message.empty() ? "" : "; ") + n + " on red at t=" + fmt(e.at("t").get<double>()) + " s";
            }
        }
        if (out.ok) out.message = "no vehicle crossed a stop line on red";
        return out;
    }

    const RunData& run_;
    const Report& report_;
    Visibility vis_;
};

}  // namespace

void runCameraCoverage(const RunData& run, const Spec& scene, Report& r) {
    Visibility vis(run);
    const std::set<std::string> referenced = referencedEntities(scene);
    std::vector<std::string> listed, scenery;
    size_t unlisted = 0, movingListed = 0;
    for (const auto& [id, e] : run.entities()) {
        if (id == "ground" || e.kind == "joint" || e.kind == "traffic_signal") continue;
        if (referenced.count(id)) listed.push_back(id);
        else if (!e.moving) scenery.push_back(id);
        else if (movingListed++ < kMaxListedMoving) listed.push_back(id);
        else ++unlisted;
    }
    const int stride = std::max(1, run.frames() / 48);
    const int last = run.frames() - 1;
    for (const CameraTrack& cam : run.cameras()) {
        Table t{"Camera " + cam.id + ": what it sees", "",
                {"entity", "in view", "visible when in view", "mean screen area", "timeline"}, {}};
        const Pose &p0 = cam.poses.front(), &p1 = cam.poses.back();
        std::ostringstream note;
        note << "Camera at " << where(p0.p) << ", fov " << fmt(cam.k.fovDeg) << " deg, " << cam.k.width << "x"
             << cam.k.height << ".";
        if (!cam.parent.empty()) note << " Rides on " << cam.parent << " (never an occluder, not listed).";
        // A lens inside a solid (a CCTV camera at the very top of its pole) sees nothing but that solid.
        for (int f = 0; f < run.frames(); f += std::max(1, run.frames() / 24)) {
            const Part* inside = nullptr;
            for (const Part& part : run.parts())
                if (part.entity != cam.parent && part.sdf(f, cam.poses[size_t(f)].p) < 0.05) inside = &part;
            if (!inside) continue;
            r.add({"camera_inside", Severity::Fail, cam.id,
                   cam.id + " is inside or touching " + inside->entity + " at t=" + fmt(run.time(f)) +
                       " s, so that body blocks its view",
                   "move the camera out of it (at least 0.2 m clear of a pole top or a wall)"});
            break;
        }
        if (run.hasGround()) {
            note << " Image corners on the ground at t=0: " << footprint(cam, p0, run.groundZ()) << ".";
            if ((p1.p - p0.p).norm() > 0.5 || p1.q.axisZ().dot(p0.q.axisZ()) < std::cos(deg2rad(5)))
                note << " At t=" << fmt(run.time(last)) << " s (camera at " << where(p1.p)
                     << "): " << footprint(cam, p1, run.groundZ()) << ".";
        }
        t.note = note.str();
        for (const std::string& id : listed) {
            if (id == cam.parent) continue;
            std::vector<bool> on;
            int n = 0, inView = 0;
            double visSum = 0, areaSum = 0;
            for (int f = 0; f < run.frames(); f += stride) {
                const View& v = vis.at(cam, id, f);
                ++n;
                on.push_back(v.visible > 0);
                if (v.visible > 0) {
                    ++inView;
                    visSum += v.visible / v.inFrame;
                    areaSum += v.area;
                }
            }
            std::vector<bool> full(size_t(run.frames()));  // the sampled timeline, held over each stride
            for (int f = 0; f < run.frames(); ++f) full[size_t(f)] = on[size_t(f / stride)];
            t.rows.push_back({id, percent(double(inView) / n), inView ? percent(visSum / inView) : "-",
                              inView ? percent(areaSum / inView) : "-", timeline(run, full)});
        }
        // Static scenery: in frame (projection only, no occlusion) at the start, middle or end?
        std::vector<std::string> framed, unframed;
        for (const std::string& id : scenery) {
            bool any = false;
            for (int f : {0, last / 2, last})
                for (const Vec3& p : run.samples(id, f, 60))
                    any = any || inClip(project(cam.poses[size_t(f)], cam.k, p), cam.k);
            (any ? framed : unframed).push_back(id);
        }
        const auto names = [](const std::vector<std::string>& ids) {
            std::string s;
            for (size_t i = 0; i < ids.size() && i < 30; ++i) s += (i ? ", " : "") + ids[i];
            return ids.size() > 30 ? s + " and " + std::to_string(ids.size() - 30) + " more" : s;
        };
        if (!scenery.empty())
            t.note += "\n\nStatic scenery in frame at t=0, mid or end (occlusion not tested): " +
                      (framed.empty() ? std::string("none") : names(framed)) + "; out of frame: " +
                      (unframed.empty() ? std::string("none") : names(unframed)) + ".";
        if (unlisted > 0)
            t.note += "\n\n" + std::to_string(unlisted) + " more moving entities not listed; name one in an expectation "
                      "to see it.";
        r.table(t);
    }
}

void runExpectations(const RunData& run, const Spec& scene, Report& r) {
    Judge judge(run, r);
    std::set<std::string> camerasCovered, ids;
    bool physical = false;
    int index = 0;
    for (const Spec& s : scene.items("expect")) {
        ++index;
        const std::string type = s.oneOf("type", {"event", "no_event", "event_order", "in_view", "enters_view", "leaves_view",
                                                  "event_in_view", "reaches", "speed", "at_rest", "min_gap", "upright",
                                                  "spin", "turns", "deformation", "particles_in_region", "obeys_signals"});
        const std::string id = s.str("id", type + std::to_string(index));
        if (!ids.insert(id).second) s["id"].fail("duplicate expectation id '" + id + "'");
        if (s.has("camera")) camerasCovered.insert(s.str("camera"));
        else physical = true;
        const Outcome o = judge(type, s);
        s.rejectUnknownKeys();
        r.add({"expect_" + type, o.ok ? Severity::Pass : Severity::Fail, id, o.message, ""});
    }
    std::vector<std::string> missing;
    for (const CameraTrack& c : run.cameras())
        if (!camerasCovered.count(c.id)) missing.push_back(c.id);
    if (!missing.empty() || !physical) {
        std::string m = physical ? "" : "no physical outcome";
        for (const auto& c : missing) m += (m.empty() ? "" : ", ") + std::string("nothing for camera ") + c;
        r.add({"expect_coverage", Severity::Fail, "", "expect has " + m, "add expectations (see the iteration guide)"});
    }
}

}  // namespace p2s
