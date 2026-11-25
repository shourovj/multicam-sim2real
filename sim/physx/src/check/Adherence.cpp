#include "check/Adherence.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>

#include "check/Formula.h"
#include "check/Visibility.h"

#ifndef P2S_SPEC_DIR
#define P2S_SPEC_DIR "spec"
#endif

namespace p2s {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// ---------------------------------------------------------------------------------------------------------------
// The vocabulary: predicates a spec formula may use, with their arguments.
//   o: an object (or a part, where noted), c: a camera, n: a number, w: a colour word; '?' marks optional ones.
// ---------------------------------------------------------------------------------------------------------------

struct PredicateInfo {
    std::string args;                 // e.g. "on?n": object, number, optional number
    std::set<std::string> keys;       // keyword arguments allowed
    bool partOk = false;              // the first object argument may be a part (sedan.front)
    bool secondPartOk = false;        // the second may be
    std::string help;
};

const std::map<std::string, PredicateInfo>& predicates() {
    static const std::map<std::string, PredicateInfo> table = {
        {"exists", {"o", {}, false, false, "the object is in the scene"}},
        {"moving", {"o?n", {}, false, false, "speed above min_mps (0.3)"}},
        {"stationary", {"o?n", {}, false, false, "speed below max_mps (0.1)"}},
        {"speed", {"onn", {}, false, false, "speed within [lo, hi] m/s"}},
        {"touches", {"oo", {}, false, false, "the two are in contact (from the first to the last touch)"}},
        {"hits", {"oo", {}, false, false, "the moment of first contact (0.25 s from it)"}},
        {"near", {"oon", {}, false, false, "surface gap at most n m"}},
        {"upright", {"o?n", {}, false, false, "tilt below n deg (30)"}},
        {"tipped", {"o?n", {}, false, false, "tilt above n deg (30)"}},
        {"airborne", {"o?n", {}, false, false, "lowest point more than n m (0.05) above the ground"}},
        {"turned", {"on", {"since"}, false, false, "heading changed by at least n deg since `since` (an atom or seconds; default t = 0)"}},
        {"moved", {"on", {"since"}, false, false, "displaced at least n m since `since`"}},
        {"slid_sideways", {"on", {"since"}, false, false, "displaced at least n m sideways (across its heading at `since`)"}},
        {"moving_away", {"oo?n", {}, false, false, "a moves away from b faster than n m/s (0.2)"}},
        {"approaching", {"oo?n", {}, false, false, "a closes on b faster than n m/s (0.2)"}},
        {"deformed", {"on", {}, true, false, "permanent crush (90th percentile of the soft bodies' vertices) at least n m"}},
        {"spins", {"on", {}, false, false, "rotation rate at least n rpm"}},
        {"crosses_on", {"ow", {}, false, false, "the vehicle passes its stop line while its light shows the colour"}},
        {"runs_red", {"o", {}, false, false, "the vehicle passes its stop line on red"}},
        {"signal_is", {"ow", {}, false, false, "the traffic light shows the colour"}},
        {"emitting", {"o", {}, false, false, "particles are being added (their count grows)"}},
        {"on_ground", {"o?n", {}, false, false, "at least that share (0.5) of the particles lies on the ground"}},
        {"leaks_from", {"oo", {}, false, true, "the particles are emitted from that object or part (from its surface)"}},
        {"fills", {"oo?n", {}, false, false, "at least that share (0.3) of the particles is inside the container's bounds"}},
        {"in_front_of", {"oo", {}, false, false, "a is ahead of b along b's heading"}},
        {"behind", {"oo", {}, false, false, "a is behind b"}},
        {"left_of", {"oo", {}, false, false, "a is on b's left (b's frame)"}},
        {"right_of", {"oo", {}, false, false, "a is on b's right"}},
        {"above", {"oo", {}, false, false, "a's lowest point is above b's highest, overlapping it in plan"}},
        {"on", {"oo", {}, false, false, "a rests on top of b"}},
        {"inside", {"oo", {}, false, false, "a's centre is inside b's bounds"}},
        {"visible", {"co?n", {}, false, false, "the camera sees the object, at least n px (24) tall on screen"}},
        {"in_frame", {"co", {}, false, false, "the object is inside the camera's frame (occlusion ignored)"}},
    };
    return table;
}

const std::set<std::string>& modes() {
    static const std::set<std::string> m = {"existence", "appearance", "spatial", "action", "camera", "consistency"};
    return m;
}

const std::set<std::string>& regions() {
    static const std::set<std::string> r = {"front", "rear", "left", "right", "side", "top", "bottom", "any"};
    return r;
}

const std::set<std::string>& cameraTypes() {
    static const std::set<std::string> t = {"fixed", "attached", "handheld", "orbit", "follow", "any"};
    return t;
}

// ---------------------------------------------------------------------------------------------------------------
// Classes and colours (spec/classes.json)
// ---------------------------------------------------------------------------------------------------------------

struct Classes {
    Json doc;
    static const Classes& get() {
        static const Classes c = [] {
            Classes out;
            const std::string path = std::string(P2S_SPEC_DIR) + "/classes.json";
            std::ifstream in(path);
            if (!in) throw PromptSpecError("cannot read " + path);
            out.doc = Json::parse(in);
            return out;
        }();
        return c;
    }
    bool has(const std::string& cls) const { return doc.at("classes").contains(cls); }
    const Json& at(const std::string& cls) const { return doc.at("classes").at(cls); }
    // The class an appearance word names ("delivery van" -> van), preferring an exact class id.
    std::vector<std::string> classesNamed(std::string word) const {
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        std::vector<std::string> out;
        for (const auto& [id, c] : doc.at("classes").items()) {
            if (id == word) { out.insert(out.begin(), id); continue; }
            for (const Json& n : c.value("names", Json::array()))
                if (n.get<std::string>() == word) out.push_back(id);
        }
        if (out.empty())  // a longer label: any name it contains as a whole word ("white delivery van")
            for (const auto& [id, c] : doc.at("classes").items())
                for (const Json& n : c.value("names", Json::array())) {
                    const std::string s = n.get<std::string>();
                    const size_t at = (" " + word + " ").find(" " + s + " ");
                    if (at != std::string::npos) { out.push_back(id); break; }
                }
        return out;
    }
    std::optional<std::string> colorFamily(std::string word) const {
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        const Json& c = doc.at("colors");
        if (c.at("families").contains(word)) return word;
        if (c.at("aliases").contains(word)) return c.at("aliases").at(word).get<std::string>();
        return std::nullopt;
    }
    std::vector<std::string> nearFamilies(const std::string& family) const {
        const Json& n = doc.at("colors").at("near");
        std::vector<std::string> out;
        if (n.contains(family))
            for (const Json& f : n.at(family)) out.push_back(f);
        return out;
    }
};

Vec3 srgbToLab(const Vec3& c) {
    const auto lin = [](double v) { return v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4); };
    const double r = lin(c.x), g = lin(c.y), b = lin(c.z);
    double X = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047, Y = 0.2126 * r + 0.7152 * g + 0.0722 * b,
           Z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
    const auto f = [](double t) { return t > 0.008856 ? std::cbrt(t) : 7.787 * t + 16.0 / 116; };
    X = f(X), Y = f(Y), Z = f(Z);
    return {116 * Y - 16, 500 * (X - Y), 200 * (Y - Z)};
}

// The colour family of an appearance colour: a word, or the family of the nearest prototype in CIELAB.
std::optional<std::string> familyOf(const Json& color) {
    const Classes& cl = Classes::get();
    if (color.is_string()) return cl.colorFamily(color.get<std::string>());
    if (!color.is_array() || color.size() != 3) return std::nullopt;
    const Vec3 lab = srgbToLab({color[0].get<double>(), color[1].get<double>(), color[2].get<double>()});
    double best = kInf;
    std::string fam;
    for (const auto& [name, protos] : cl.doc.at("colors").at("families").items())
        for (const Json& p : protos)
            if (const double d = (srgbToLab({p[0], p[1], p[2]}) - lab).norm(); d < best) best = d, fam = name;
    return fam;
}

bool colorMatches(const std::string& wanted, const std::string& got) {
    if (wanted == got) return true;
    const auto near = Classes::get().nearFamilies(wanted);
    return std::find(near.begin(), near.end(), got) != near.end();
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

std::vector<std::string> words(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : lower(s) + " ") {
        if (std::isalnum(static_cast<unsigned char>(c))) cur += c;
        else if (!cur.empty()) out.push_back(cur), cur.clear();
    }
    return out;
}

std::string fmtTimes(const RunData& run, const std::vector<double>& v) {
    std::ostringstream s;
    int start = -1, runs = 0;
    for (int f = 0; f <= int(v.size()); ++f) {
        const bool on = f < int(v.size()) && v[size_t(f)] > 0;
        if (on && start < 0) start = f;
        if (!on && start >= 0) {
            if (runs++ < 4) s << (runs > 1 ? ", " : "") << fmt(run.time(start), 3) << "-" << fmt(run.time(f - 1), 3) << " s";
            start = -1;
        }
    }
    if (runs > 4) s << " and " << runs - 4 << " more";
    return s.str();
}

// ---------------------------------------------------------------------------------------------------------------
// Spec structure
// ---------------------------------------------------------------------------------------------------------------

struct ObjectSpec {
    std::string name, cls, color, span;
    int count = 1;
};

struct PartSpec {
    std::string name, of, region, span;
};

struct CameraSpec {
    std::string name, type, on, mountedOn, span;
    double minHeight = -kInf;
};

struct Requirement {
    std::string id, mode, text;
    FormulaPtr formula;
    bool implied = false;  // not said in the prompt but follows from it (a crash ends the drive): a warning
};

struct ParsedSpec {
    std::string prompt;
    std::vector<ObjectSpec> objects;
    std::vector<PartSpec> parts;
    std::vector<CameraSpec> cameras;
    std::vector<Requirement> requirements;
    std::vector<std::string> notSimulated;
};

// Parses and validates; every problem goes to `errors` (with a JSON path), so a linter can list them all.
ParsedSpec parseSpec(const Json& j, std::vector<std::string>& errors) {
    ParsedSpec out;
    const auto err = [&](const std::string& path, const std::string& m) { errors.push_back("spec." + path + ": " + m); };
    if (!j.is_object()) { errors.push_back("spec: expected a JSON object"); return out; }
    const std::set<std::string> top = {"format", "version", "prompt", "setting", "objects", "parts", "cameras", "requirements",
                                       "not_simulated", "writer", "notes"};
    for (const auto& [k, v] : j.items())
        if (!top.count(k)) err(k, "unknown key");
    if (j.value("format", "") != "p2s-spec") err("format", "must be \"p2s-spec\"");
    out.prompt = j.value("prompt", "");
    if (out.prompt.empty()) err("prompt", "missing");
    const Classes& cl = Classes::get();
    std::set<std::string> names;
    const auto str = [&](const Json& o, const char* k, const std::string& path, bool required) {
        if (!o.contains(k)) { if (required) err(path + "." + k, "missing"); return std::string(); }
        if (!o.at(k).is_string()) { err(path + "." + k, "expected a string"); return std::string(); }
        return o.at(k).get<std::string>();
    };
    const auto checkKeys = [&](const Json& o, const std::string& path, const std::set<std::string>& allowed) {
        for (const auto& [k, v] : o.items())
            if (!allowed.count(k)) err(path + "." + k, "unknown key");
    };
    const auto validName = [&](const std::string& n, const std::string& path) {
        if (n.empty() || n.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_") != std::string::npos || std::isdigit(n[0]))
            err(path, "names are lower_snake_case words: '" + n + "'");
        if (!names.insert(n).second) err(path, "duplicate name '" + n + "'");
    };
    const Json objects = j.value("objects", Json::array());
    for (size_t i = 0; i < objects.size(); ++i) {
        const Json& o = objects[i];
        const std::string path = "objects[" + std::to_string(i) + "]";
        if (!o.is_object()) { err(path, "expected an object"); continue; }
        checkKeys(o, path, {"name", "class", "count", "color", "span"});
        ObjectSpec s{str(o, "name", path, true), str(o, "class", path, true), str(o, "color", path, false), str(o, "span", path, false)};
        validName(s.name, path + ".name");
        if (!s.cls.empty() && !cl.has(s.cls)) err(path + ".class", "unknown class '" + s.cls + "' (spec/classes.json)");
        if (!s.color.empty() && !cl.colorFamily(s.color)) err(path + ".color", "unknown colour word '" + s.color + "' (spec/classes.json colors)");
        if (o.contains("count")) {
            if (!o["count"].is_number_integer() || o["count"].get<int>() < 1) err(path + ".count", "expected an integer >= 1");
            else s.count = o["count"].get<int>();
        }
        out.objects.push_back(s);
    }
    std::set<std::string> objectNames;
    for (const auto& o : out.objects) objectNames.insert(o.name);
    const Json parts = j.value("parts", Json::array());
    for (size_t i = 0; i < parts.size(); ++i) {
        const Json& p = parts[i];
        const std::string path = "parts[" + std::to_string(i) + "]";
        if (!p.is_object()) { err(path, "expected an object"); continue; }
        checkKeys(p, path, {"name", "of", "region", "span"});
        PartSpec s{str(p, "name", path, true), str(p, "of", path, true), str(p, "region", path, true), str(p, "span", path, false)};
        if (!objectNames.count(s.of)) err(path + ".of", "no object '" + s.of + "'");
        if (s.name.rfind(s.of + ".", 0) != 0) err(path + ".name", "a part is named <object>.<part>, e.g. '" + s.of + ".front'");
        if (!names.insert(s.name).second) err(path + ".name", "duplicate name '" + s.name + "'");
        if (!regions().count(s.region)) err(path + ".region", "one of front rear left right side top bottom any");
        out.parts.push_back(s);
    }
    const Json cams = j.value("cameras", Json::array());
    for (size_t i = 0; i < cams.size(); ++i) {
        const Json& c = cams[i];
        const std::string path = "cameras[" + std::to_string(i) + "]";
        if (!c.is_object()) { err(path, "expected an object"); continue; }
        checkKeys(c, path, {"name", "type", "on", "mounted_on", "min_height_m", "span"});
        CameraSpec s{str(c, "name", path, true), str(c, "type", path, false), str(c, "on", path, false),
                     str(c, "mounted_on", path, false), str(c, "span", path, false)};
        if (s.type.empty()) s.type = "any";
        validName(s.name, path + ".name");
        if (!cameraTypes().count(s.type)) err(path + ".type", "one of fixed attached handheld orbit follow any");
        for (const std::string& ref : {s.on, s.mountedOn})
            if (!ref.empty() && !objectNames.count(ref)) err(path, "no object '" + ref + "'");
        if (c.contains("min_height_m")) {
            if (!c["min_height_m"].is_number()) err(path + ".min_height_m", "expected a number");
            else s.minHeight = c["min_height_m"].get<double>();
        }
        out.cameras.push_back(s);
    }
    std::set<std::string> cameraNames;
    for (const auto& c : out.cameras) cameraNames.insert(c.name);
    std::set<std::string> partNames;
    for (const auto& p : out.parts) partNames.insert(p.name);
    std::set<std::string> ids;
    const Json reqs = j.value("requirements", Json::array());
    for (size_t i = 0; i < reqs.size(); ++i) {
        const Json& r = reqs[i];
        const std::string path = "requirements[" + std::to_string(i) + "]";
        if (!r.is_object()) { err(path, "expected an object"); continue; }
        checkKeys(r, path, {"id", "mode", "text", "formula", "span", "implied"});
        Requirement q{str(r, "id", path, true), str(r, "mode", path, true), str(r, "text", path, true), nullptr};
        if (r.contains("implied")) {
            if (!r["implied"].is_boolean()) err(path + ".implied", "expected true or false");
            else q.implied = r["implied"].get<bool>();
        }
        if (!ids.insert(q.id).second) err(path + ".id", "duplicate id '" + q.id + "'");
        if (!q.mode.empty() && !modes().count(q.mode)) err(path + ".mode", "one of existence appearance spatial action camera consistency");
        const std::string src = str(r, "formula", path, true);
        try {
            q.formula = parseFormula(src);
        } catch (const std::exception& e) {
            err(path + ".formula", e.what());
            continue;
        }
        forEachAtom(*q.formula, [&](const Formula& a) {
            const auto it = predicates().find(a.pred);
            if (it == predicates().end()) {
                std::string list;
                for (const auto& [n, p] : predicates()) list += (list.empty() ? "" : " ") + n;
                err(path + ".formula", "unknown predicate '" + a.pred + "' (use one of: " + list + ")");
                return;
            }
            const PredicateInfo& info = it->second;
            size_t pos = 0, objectArgs = 0;
            for (const Arg& g : a.args) {
                if (!g.key.empty()) {
                    if (!info.keys.count(g.key)) err(path + ".formula", a.pred + " has no argument '" + g.key + "='");
                    else if (g.key == "since" && !g.atom && !g.isNumber) err(path + ".formula", "since= takes an atom or seconds");
                    continue;
                }
                std::string kinds;  // the argument kinds in order, '?' dropped
                for (char ch : info.args) if (ch != '?') kinds += ch;
                if (pos >= kinds.size()) { err(path + ".formula", a.text() + ": too many arguments"); break; }
                const char k = kinds[pos];
                const std::string where = path + ".formula";
                if (k == 'n' && !g.isNumber) err(where, a.text() + ": argument " + std::to_string(pos + 1) + " must be a number");
                if (k == 'w' && (g.name.empty() || !cl.colorFamily(g.name))) err(where, a.text() + ": argument " + std::to_string(pos + 1) + " must be a colour word");
                if (k == 'c' && !cameraNames.count(g.name)) err(where, a.text() + ": no camera '" + g.name + "' in spec.cameras");
                if (k == 'o') {
                    const bool partAllowed = (objectArgs == 0 && info.partOk) || (objectArgs == 1 && info.secondPartOk);
                    if (g.name.empty()) err(where, a.text() + ": argument " + std::to_string(pos + 1) + " must be an object name");
                    else if (partNames.count(g.name)) { if (!partAllowed) err(where, a.text() + ": a part is not allowed here; name its object"); }
                    else if (!objectNames.count(g.name)) err(where, a.text() + ": no object '" + g.name + "' in spec.objects");
                    ++objectArgs;
                }
                ++pos;
            }
            size_t required = 0;
            for (size_t c = 0; c < info.args.size(); ++c)
                if (info.args[c] != '?' && (c == 0 || info.args[c - 1] != '?')) ++required;
            if (pos < required) err(path + ".formula", a.text() + ": needs " + std::to_string(required) + " arguments (" + info.help + ")");
        });
        out.requirements.push_back(q);
    }
    for (const Json& n : j.value("not_simulated", Json::array()))
        out.notSimulated.push_back(n.is_string() ? n.get<std::string>() : n.value("text", n.dump()));
    return out;
}

// ---------------------------------------------------------------------------------------------------------------
// Entities as the checker sees them
// ---------------------------------------------------------------------------------------------------------------

struct EntityFacts {
    std::string id, kind;
    Vec3 size;          // length, width, height in its own frame at t = 0 (incl. mounted soft bodies)
    Json appearance;    // from S
    std::string preset; // particles
    bool hostedSoft = false;
};

class Scene {
public:
    Scene(const RunData& run, const Spec& scene) : run_(run) {
        const Json& s = scene.raw();
        for (const auto& [key, list] : s.items()) {
            if (!list.is_array()) continue;
            for (const Json& e : list)
                if (e.is_object() && e.contains("id") && e["id"].is_string()) raw_[e["id"].get<std::string>()] = e;
        }
        for (const auto& [id, info] : run.entities()) {
            if (id == "ground" || info.kind == "joint") continue;
            EntityFacts f{id, info.kind, {}, Json(), "", false};
            if (raw_.count(id) && raw_[id].contains("appearance")) f.appearance = raw_[id]["appearance"];
            if (info.kind == "particles") f.preset = info.desc.value("preset", "");
            if (info.kind == "soft_body") f.hostedSoft = !info.desc.value("host", "").empty();
            f.size = sizeOf(id, info);
            facts_[id] = f;
        }
    }

    const std::map<std::string, EntityFacts>& facts() const { return facts_; }
    const Json* raw(const std::string& id) const { const auto it = raw_.find(id); return it == raw_.end() ? nullptr : &it->second; }

    // The pose of an entity's own frame (its main part) at a frame; identity at its centroid for point sets.
    std::optional<Pose> frameOf(const std::string& id, int f) const {
        if (const auto parts = run_.partsOf(id); !parts.empty()) return parts.front().get().poses[size_t(f)];
        if (const auto p = run_.position(id, f)) return Pose{*p, Quat{}};
        return std::nullopt;
    }

private:
    Vec3 sizeOf(const std::string& id, const EntityInfo& info) const {
        if (info.kind == "walker") {
            const double h = info.desc.value("height_m", 1.75);
            return {0.5, 0.5, h};
        }
        // Its rigid parts and the soft bodies mounted on it (a crumple zone is part of a car's length); not a flag on
        // a pole or a liquid in a glass.
        std::vector<Vec3> pts;
        for (const Part& p : run_.partsOf(id))
            for (const Vec3& s : p.samples) pts.push_back(p.poses[0].transform(s));
        for (const PointSet& ps : run_.pointSets())
            if ((ps.entity == id || (ps.host == id && ps.kind == "soft_body")) && !ps.frames.empty())
                pts.insert(pts.end(), ps.frames[0].begin(), ps.frames[0].end());
        if (pts.empty()) return {};
        const auto fr = frameOf(id, 0);
        Aabb box;
        for (const Vec3& p : pts) box.add(fr ? fr->transformInv(p) : p);
        Vec3 e = box.extent();
        // Vehicles have a forward axis; other bodies report their longest horizontal extent as the length.
        if (info.kind != "vehicle" && info.kind != "tracked_vehicle" && e.y > e.x) std::swap(e.x, e.y);
        return e;
    }

    const RunData& run_;
    std::map<std::string, Json> raw_;
    std::map<std::string, EntityFacts> facts_;
};

// ---------------------------------------------------------------------------------------------------------------
// Binding
// ---------------------------------------------------------------------------------------------------------------

struct Candidate {
    double score = 0;
    std::vector<std::string> notes;
};

double sizePenalty(const Json& range, const Vec3& size, std::vector<std::string>* problems) {
    if (!range.is_array()) return 0;
    static const char* dims[] = {"length", "width", "height"};
    double pen = 0;
    for (int i = 0; i < 3 && i < int(range.size()); ++i) {
        if (!range[size_t(i)].is_array()) continue;
        const double lo = range[size_t(i)][0], hi = range[size_t(i)][1], v = size[i];
        if (v <= 0) continue;
        double off = 0;
        if (v < lo) off = std::log(lo / v);
        if (v > hi) off = std::log(v / hi);
        pen += off;
        if (off > std::log(1.15) && problems)
            problems->push_back(std::string(dims[i]) + " " + fmt(v) + " m (a " + "class " + dims[i] + " is " + fmt(lo) + "-" + fmt(hi) + " m)");
    }
    return pen;
}

Candidate scoreCandidate(const ObjectSpec& o, const EntityFacts& e) {
    const Classes& cl = Classes::get();
    Candidate c;
    const Json& cls = cl.at(o.cls);
    const Json kinds = cls.value("kinds", Json::array());
    if (std::find(kinds.begin(), kinds.end(), e.kind) == kinds.end() || e.hostedSoft) { c.score = -kInf; return c; }
    if (cls.contains("preset")) {
        const Json pr = cls["preset"];
        if (std::find(pr.begin(), pr.end(), e.preset) == pr.end()) c.score -= 2;
    }
    c.score -= 2 * sizePenalty(cls.value("size_m", Json()), e.size, nullptr);
    if (e.appearance.is_object()) {
        if (e.appearance.contains("class")) {
            const auto named = cl.classesNamed(e.appearance["class"].get<std::string>());
            const bool same = std::find(named.begin(), named.end(), o.cls) != named.end();
            c.score += same ? 3 : (named.empty() ? 0 : -3);
        }
        if (!o.color.empty() && e.appearance.contains("color"))
            if (const auto fam = familyOf(e.appearance["color"]))
                c.score += colorMatches(*cl.colorFamily(o.color), *fam) ? 1.5 : -1.5;
    }
    // Words of the prompt's name and class in the entity id (car_red, van, ped_corner).
    std::set<std::string> want;
    for (const auto& w : words(o.name)) want.insert(w);
    for (const auto& w : words(o.cls)) want.insert(w);
    for (const Json& n : cls.value("names", Json::array()))
        for (const auto& w : words(n.get<std::string>())) want.insert(w);
    if (!o.color.empty()) want.insert(lower(o.color));
    for (const auto& w : words(e.id))
        if (want.count(w) || (w.size() >= 3 && std::any_of(want.begin(), want.end(), [&](const std::string& x) { return x.rfind(w, 0) == 0; })))
            c.score += 1;
    return c;
}

struct Binding {
    std::map<std::string, std::vector<std::string>> objects;   // object name -> entity ids
    std::map<std::string, std::string> cameras;                 // camera name -> camera id
};

Binding bindObjects(const ParsedSpec& spec, const Scene& scene) {
    Binding b;
    struct Pair { double score; size_t obj; std::string ent; };
    std::vector<Pair> pairs;
    for (size_t i = 0; i < spec.objects.size(); ++i)
        for (const auto& [id, e] : scene.facts())
            if (const double s = scoreCandidate(spec.objects[i], e).score; s > -kInf) pairs.push_back({s, i, id});
    std::stable_sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& c) { return a.score > c.score; });
    std::set<std::string> used;
    for (const Pair& p : pairs) {
        const ObjectSpec& o = spec.objects[p.obj];
        auto& got = b.objects[o.name];
        if (int(got.size()) >= o.count || used.count(p.ent)) continue;
        got.push_back(p.ent);
        used.insert(p.ent);
    }
    return b;
}

// ---------------------------------------------------------------------------------------------------------------
// Grounding: the value of every atom at every frame
// ---------------------------------------------------------------------------------------------------------------

struct AtomValue {
    std::vector<double> v;
    std::string note;  // what was measured, in words
};

class Grounder {
public:
    Grounder(const RunData& run, const Scene& scene, const ParsedSpec& spec, const Binding& b)
        : run_(run), scene_(scene), spec_(spec), b_(b), vis_(run), n_(size_t(run.frames())) {}

    const AtomValue& operator()(const Formula& a) {
        const std::string key = a.text();
        if (const auto it = cache_.find(key); it != cache_.end()) return it->second;
        AtomValue out = evaluate(a);
        if (out.v.size() != n_) out.v.assign(n_, -1);
        return cache_[key] = std::move(out);
    }

    // Region of a part in its host's frame, and the soft bodies mounted there.
    struct PartBinding {
        std::string host;
        std::string region;
        std::vector<std::string> soft;
    };
    std::optional<PartBinding> part(const std::string& name) const {
        for (const PartSpec& p : spec_.parts) {
            if (p.name != name) continue;
            const auto it = b_.objects.find(p.of);
            if (it == b_.objects.end() || it->second.empty()) return std::nullopt;
            PartBinding pb{it->second.front(), p.region, {}};
            for (const PointSet& ps : run_.pointSets())
                if (ps.kind == "soft_body" && ps.host == pb.host && inRegion(pb.host, centroid(ps.rest), p.region)) pb.soft.push_back(ps.entity);
            return pb;
        }
        return std::nullopt;
    }

    // Is a point (in the host's frame) in that region of the host?
    bool inRegion(const std::string& host, const Vec3& local, const std::string& region) const {
        const auto& f = scene_.facts().at(host);
        const double hx = std::max(f.size.x / 2, 0.05), hy = std::max(f.size.y / 2, 0.05), h = std::max(f.size.z, 0.05);
        const double nx = local.x / hx, ny = local.y / hy, nz = local.z / h;
        if (region == "front") return nx > 0.35;
        if (region == "rear") return nx < -0.35;
        if (region == "left") return ny > 0.35;
        if (region == "right") return ny < -0.35;
        if (region == "side") return std::fabs(ny) > 0.35 && std::fabs(nx) < 0.9;
        if (region == "top") return nz > 0.6;
        if (region == "bottom") return nz < 0.3;
        return true;
    }

    static Vec3 centroid(const std::vector<Vec3>& pts) {
        Vec3 c;
        for (const Vec3& p : pts) c += p;
        return pts.empty() ? c : c / double(pts.size());
    }

private:
    std::vector<std::string> ents(const std::string& name) const {
        if (const auto it = b_.objects.find(name); it != b_.objects.end()) return it->second;
        if (const auto pb = part(name)) return {pb->host};
        return {};
    }

    const Arg* positional(const Formula& a, size_t i) const {
        size_t k = 0;
        for (const Arg& g : a.args)
            if (g.key.empty() && k++ == i) return &g;
        return nullptr;
    }
    // The i-th numeric positional argument.
    double number(const Formula& a, size_t i, double fallback) const {
        size_t k = 0;
        for (const Arg& g : a.args)
            if (g.key.empty() && g.isNumber && k++ == i) return g.number;
        return fallback;
    }
    std::string name(const Formula& a, size_t i) const {
        const Arg* g = positional(a, i);
        return g ? g->name : "";
    }
    const Arg* keyword(const Formula& a, const std::string& key) const {
        for (const Arg& g : a.args)
            if (g.key == key) return &g;
        return nullptr;
    }

    // The anchor frame of since=: the first frame its atom holds, or the frame at its seconds; 0 without one.
    std::optional<int> anchor(const Formula& a, std::string& note) {
        const Arg* g = keyword(a, "since");
        if (!g) return 0;
        if (g->isNumber) return run_.frameAt(g->number);
        const AtomValue& v = (*this)(*g->atom);
        for (size_t f = 0; f < n_; ++f)
            if (v.v[f] > 0) return int(f);
        note = "since: " + g->atom->text() + " never holds";
        return std::nullopt;
    }

    std::vector<Vec3> velocity(const std::string& id) const {
        const auto& info = run_.entity(id);
        if (info.velocity.size() == n_) return info.velocity;
        std::vector<Vec3> v(n_);
        for (size_t f = 1; f < n_; ++f) {
            const auto a = run_.position(id, int(f - 1)), b = run_.position(id, int(f));
            if (a && b) v[f] = (*b - *a) * run_.fps();
        }
        if (n_ > 1) v[0] = v[1];
        return v;
    }

    double heading(const std::string& id, int f) const {
        const auto fr = scene_.frameOf(id, f);
        if (!fr) return 0;
        const Vec3 x = fr->q.axisX();
        return std::atan2(x.y, x.x);
    }

    Aabb bounds(const std::string& id, int f) const {
        Aabb box;
        for (const Vec3& p : run_.samples(id, f, 400)) box.add(p);
        return box;
    }

    // Surface gap between two entities at a frame: sample points against the other's rigid shapes, both ways.
    double gap(const std::string& a, const std::string& b, int f) const {
        const Aabb ba = bounds(a, f), bb = bounds(b, f);
        if (ba.empty() || bb.empty()) return kInf;
        const double boxGap = ((bb.lo - ba.hi).cwiseMax(ba.lo - bb.hi).cwiseMax({0, 0, 0})).norm();
        if (boxGap > 1.0) return boxGap;
        const auto pa = run_.samples(a, f, 150), pb = run_.samples(b, f, 150);
        double best = kInf;
        for (const Part& p : run_.partsOf(b)) for (const Vec3& x : pa) best = std::min(best, p.sdf(f, x));
        for (const Part& p : run_.partsOf(a)) for (const Vec3& y : pb) best = std::min(best, p.sdf(f, y));
        if (best == kInf)
            for (const Vec3& x : pa) for (const Vec3& y : pb) best = std::min(best, (x - y).norm());
        return std::max(best, boxGap > 0 ? boxGap : 0.0);
    }

    // Contact records between two entities, counting what is mounted on each (a crumple zone reports as its host).
    std::vector<std::pair<double, double>> contacts(const std::string& a, const std::string& b) const {
        const auto family = [&](const std::string& id) {
            std::set<std::string> s{id};
            for (const PointSet& p : run_.pointSets()) if (p.host == id) s.insert(p.entity);
            return s;
        };
        const auto fa = family(a), fb = family(b);
        std::vector<std::pair<double, double>> out;
        for (const Json& c : run_.x().value("contacts", Json::array())) {
            const std::string x = c.value("a", ""), y = c.value("b", "");
            if ((fa.count(x) && fb.count(y)) || (fa.count(y) && fb.count(x)))
                out.push_back({c.value("first_t", 0.0), c.value("last_t", 0.0)});
        }
        return out;
    }

    std::vector<double> perEntity(const std::vector<std::string>& ids, const std::function<std::vector<double>(const std::string&)>& fn) const {
        std::vector<double> out(n_, 1.0);
        for (const auto& id : ids) {
            const auto v = fn(id);
            for (size_t f = 0; f < n_; ++f) out[f] = std::min(out[f], v[f]);
        }
        return out;
    }

    AtomValue unbound(const std::string& nm) const {
        return {std::vector<double>(n_, -1.0), "'" + nm + "' is not bound to any entity of the scene"};
    }

    std::string signalOf(const std::string& vehicle) const {
        const Json& d = run_.entity(vehicle).desc;
        return d.contains("driver") ? d["driver"].value("signal", "") : "";
    }

    AtomValue evaluate(const Formula& a) {
        const std::string& p = a.pred;
        const std::string o0 = name(a, 0), o1 = name(a, 1);
        AtomValue out;
        out.v.assign(n_, -1.0);
        std::ostringstream note;

        if (p == "visible" || p == "in_frame") {
            const auto cit = b_.cameras.find(o0);
            if (cit == b_.cameras.end()) return {out.v, "camera '" + o0 + "' is not bound to a scene camera"};
            const CameraTrack* cam = nullptr;
            for (const CameraTrack& c : run_.cameras()) if (c.id == cit->second) cam = &c;
            const auto ids = ents(o1);
            if (ids.empty() || !cam) return unbound(o1);
            const double px = number(a, 0, 24);
            double bestH = 0, worstShown = 0;  // worstShown: the largest unhidden share seen in any frame
            int bestF = 0;
            out.v = perEntity(ids, [&](const std::string& id) {
                std::vector<double> v(n_);
                for (size_t f = 0; f < n_; ++f) {
                    const View& w = vis_.at(*cam, id, int(f));
                    if (p == "in_frame") { v[f] = w.inFrame > 0 ? 1 : -1; continue; }
                    if (w.heightPx > bestH) bestH = w.heightPx, bestF = int(f);
                    // Seen: at least 24 px tall (video models lose smaller things) and at most 60% of what faces
                    // the camera hidden (a sliver of a car behind a bus is not "the camera shows the car").
                    const double shown = w.inFrame > 0 ? w.visible / w.inFrame : 0;
                    v[f] = w.visible <= 0 ? (w.inFrame > 0 ? -0.5 : -1.0)
                                          : std::clamp(std::min((w.heightPx - px) / px, (shown - 0.4) / 0.3), -0.4, 1.0);
                    if (w.visible > 0 && w.heightPx >= px && shown >= 0.4 && v[f] <= 0) v[f] = 1e-3;
                    worstShown = std::max(worstShown, w.visible > 0 ? shown : 0.0);
                }
                return v;
            });
            if (p == "visible")
                note << "tallest on screen " << fmt(bestH) << " px at t=" << fmt(run_.time(bestF)) << " s (needs " << fmt(px)
                     << " px); at most " << fmt(100 * worstShown) << "% of it unhidden in a frame (needs 40%)";
            out.note = note.str();
            return out;
        }

        if (p == "exists") {
            const auto ids = ents(o0);
            out.v.assign(n_, ids.empty() ? -1.0 : 1.0);
            out.note = ids.empty() ? "not in the scene" : "bound to " + ids.front();
            return out;
        }

        // Deformation and leaks accept parts.
        if (p == "deformed") {
            std::vector<std::string> soft;
            std::string host;
            if (const auto pb = part(o0)) soft = pb->soft, host = pb->host;
            else for (const auto& id : ents(o0)) {
                host = id;
                for (const PointSet& ps : run_.pointSets()) if (ps.kind == "soft_body" && (ps.host == id || ps.entity == id)) soft.push_back(ps.entity);
            }
            if (host.empty()) return unbound(o0);
            if (soft.empty())
                return {out.v, o0 + " (" + host + ") has no soft body there, so it cannot deform: mount a plastic soft body on it"};
            const double m = number(a, 0, 0.05);
            double peak = 0;
            std::vector<double> crush(n_, 0);
            for (const std::string& s : soft) {
                const auto c = crushOf(s);
                for (size_t f = 0; f < n_; ++f) crush[f] = std::max(crush[f], c[f]);
            }
            for (size_t f = 0; f < n_; ++f) {
                out.v[f] = (crush[f] - m) / std::max(m, 0.02);
                peak = std::max(peak, crush[f]);
            }
            note << "crush of ";
            for (size_t i = 0; i < soft.size(); ++i) note << (i ? ", " : "") << soft[i];
            note << " reaches " << fmt(peak) << " m (needs " << fmt(m) << " m)";
            out.note = note.str();
            return out;
        }

        if (p == "leaks_from") return leaks(a, o0, o1);

        if (p == "fills") {
            const auto C = ents(o0), P = ents(o1);
            if (C.empty()) return unbound(o0);
            if (P.empty()) return unbound(o1);
            const PointSet* ps = nullptr;
            for (const PointSet& s : run_.pointSets()) if (s.entity == P.front()) ps = &s;
            if (!ps) return {out.v, P.front() + " is not particles"};
            const double want = number(a, 0, 0.3);
            double best = 0;
            for (size_t f = 0; f < n_; ++f) {
                const auto& pts = ps->frames[f];
                const Aabb box = bounds(C.front(), int(f));
                if (pts.empty() || box.empty()) continue;
                const double share = double(std::count_if(pts.begin(), pts.end(), [&](const Vec3& q) { return box.contains(q); })) / double(pts.size());
                best = std::max(best, share);
                out.v[f] = (share - want) / 0.2;
            }
            out.note = "at most " + fmt(100 * best) + "% of " + P.front() + " inside " + C.front() + " (needs " + fmt(100 * want) + "%)";
            return out;
        }

        if (p == "crosses_on" || p == "runs_red") {
            const std::string colour = p == "runs_red" ? "red" : lower(o1);
            const auto ids = ents(o0);
            if (ids.empty()) return unbound(o0);
            std::vector<std::string> seen;
            for (const auto& id : ids) {
                if (signalOf(id).empty()) {
                    seen.push_back(id + " has no signal in its driver, so whether it crosses on " + colour +
                                   " cannot be measured: give its driver the signal, its stop_s" +
                                   (colour == "red" ? " and \"runs_red\": true" : ""));
                    continue;
                }
                bool any = false;
                for (const Json& e : run_.x().at("events")) {
                    if (e.at("name").get<std::string>() != id + ":passes_stop_line") continue;
                    any = true;
                    const std::string light = e.value("data", Json::object()).value("light", "none");
                    const int f = run_.frameAt(e.at("t").get<double>());
                    seen.push_back(id + " passed its stop line at t=" + fmt(e.at("t").get<double>()) + " s on " + light);
                    if (Classes::get().colorFamily(light) == Classes::get().colorFamily(colour))
                        for (int k = f; k <= std::min(int(n_) - 1, f + 2); ++k) out.v[size_t(k)] = 1;
                }
                if (!any) seen.push_back(id + " never passed its stop line");
            }
            for (size_t i = 0; i < seen.size(); ++i) note << (i ? "; " : "") << seen[i];
            out.note = note.str();
            return out;
        }

        if (p == "signal_is") {
            const auto ids = ents(o0);
            if (ids.empty()) return unbound(o0);
            const StreamReader sr(run_.dir(), run_.x().at("streams"));
            const std::string key = "signal/" + ids.front();
            if (!sr.has(key)) return {out.v, ids.front() + " has no signal stream"};
            const auto data = sr.read(key);
            const std::map<std::string, int> code = {{"red", 0}, {"yellow", 1}, {"orange", 1}, {"amber", 1}, {"green", 2}};
            const auto it = code.find(lower(o1));
            for (size_t f = 0; f < n_ && f < data.size(); ++f) out.v[f] = it != code.end() && int(data[f]) == it->second ? 1 : -1;
            out.note = ids.front() + " " + lower(o1) + " at " + fmtTimes(run_, out.v);
            return out;
        }

        if (p == "emitting" || p == "on_ground") {
            const auto ids = ents(o0);
            if (ids.empty()) return unbound(o0);
            const PointSet* ps = nullptr;
            for (const PointSet& s : run_.pointSets()) if (s.entity == ids.front()) ps = &s;
            if (!ps) return {out.v, ids.front() + " is not particles"};
            if (p == "emitting") {
                for (size_t f = 1; f < n_; ++f) out.v[f] = ps->frames[f].size() > ps->frames[f - 1].size() ? 1 : -1;
                out.note = ids.front() + " grows at " + fmtTimes(run_, out.v);
                return out;
            }
            const double want = number(a, 0, 0.5);
            const double z = run_.groundZ() + 2 * ps->radius + 0.05;
            double best = 0;
            for (size_t f = 0; f < n_; ++f) {
                const auto& pts = ps->frames[f];
                if (pts.empty()) continue;
                const double share = double(std::count_if(pts.begin(), pts.end(), [&](const Vec3& q) { return q.z <= z; })) / double(pts.size());
                best = std::max(best, share);
                out.v[f] = (share - want) / 0.2;
            }
            out.note = "at most " + fmt(100 * best) + "% of " + ids.front() + " on the ground (needs " + fmt(100 * want) + "%)";
            return out;
        }

        // The rest are about whole bodies.
        const auto A = ents(o0);
        if (A.empty()) return unbound(o0);
        const bool pair = p == "touches" || p == "hits" || p == "near" || p == "moving_away" || p == "approaching" ||
                          p == "in_front_of" || p == "behind" || p == "left_of" || p == "right_of" || p == "above" ||
                          p == "on" || p == "inside";
        std::vector<std::string> B;
        if (pair) {
            B = ents(o1);
            if (B.empty()) return unbound(o1);
        }

        if (p == "moving" || p == "stationary" || p == "speed") {
            const double lo = p == "speed" ? number(a, 0, 0) : p == "moving" ? number(a, 0, 0.3) : 0;
            const double hi = p == "speed" ? number(a, 1, 1e9) : p == "stationary" ? number(a, 0, 0.1) : 1e9;
            const double scale = std::max(0.3, 0.1 * (hi < 1e8 ? hi : lo));
            double vmin = kInf, vmax = 0;
            out.v = perEntity(A, [&](const std::string& id) {
                const auto vel = velocity(id);
                std::vector<double> v(n_);
                for (size_t f = 0; f < n_; ++f) {
                    const double s = vel[f].norm();
                    vmin = std::min(vmin, s), vmax = std::max(vmax, s);
                    v[f] = std::min(lo > 0 ? s - lo : kInf, hi < 1e8 ? hi - s : kInf) / scale;
                }
                return v;
            });
            note << "speed " << fmt(vmin) << "-" << fmt(vmax) << " m/s over the run";
            out.note = note.str();
            return out;
        }

        if (p == "touches" || p == "hits" || p == "near") {
            const double m = p == "near" ? number(a, 0, 1.0) : 0.0;
            double closest = kInf;
            std::set<std::string> said;
            out.v = perEntity(A, [&](const std::string& x) {
                return perEntity(B, [&](const std::string& y) {
                    std::vector<double> v(n_, -1.0);
                    const auto cs = contacts(x, y);
                    for (const auto& [t0, t1] : cs) {
                        said.insert(x + " touches " + y + " " + fmt(t0) + "-" + fmt(t1) + " s");
                        const int f0 = run_.frameAt(t0), f1 = p == "hits" ? run_.frameAt(t0 + 0.25) : run_.frameAt(t1);
                        for (int f = f0; f <= f1 && f < int(n_); ++f) v[size_t(f)] = 1;
                    }
                    const int stride = cs.empty() || p == "near" ? 1 : 1000000;
                    for (int f = 0; f < int(n_); f += std::max(1, std::min(stride, int(n_)))) {
                        if (v[size_t(f)] > 0) continue;
                        const double g = gap(x, y, f);
                        closest = std::min(closest, g);
                        v[size_t(f)] = p == "near" ? (m - g) / std::max(m, 0.5) : std::max(-1.0, -g);
                        if (p == "near" && v[size_t(f)] <= 0 && g <= m) v[size_t(f)] = 1e-3;
                    }
                    return v;
                });
            });
            int listed = 0;
            for (const auto& s : said)
                if (listed < 3) note << (listed++ ? "; " : "") << s;
            if (said.empty()) note << "no contact; closest " << (closest < kInf ? fmt(closest) + " m" : "unknown");
            else if (p == "near") note << "; closest " << fmt(std::max(0.0, closest)) << " m";
            out.note = note.str();
            return out;
        }

        if (p == "upright" || p == "tipped") {
            const double deg = number(a, 0, 30);
            double worst = 0;
            out.v = perEntity(A, [&](const std::string& id) {
                std::vector<double> v(n_);
                for (size_t f = 0; f < n_; ++f) {
                    const auto fr = scene_.frameOf(id, int(f));
                    const double tilt = fr ? rad2deg(std::acos(std::clamp(fr->q.axisZ().z, -1.0, 1.0))) : 0;
                    worst = std::max(worst, tilt);
                    v[f] = (p == "upright" ? deg - tilt : tilt - deg) / 10;
                }
                return v;
            });
            out.note = "tilt up to " + fmt(worst) + " deg";
            return out;
        }

        if (p == "airborne") {
            const double m = number(a, 0, 0.05);
            out.v = perEntity(A, [&](const std::string& id) {
                std::vector<double> v(n_);
                for (size_t f = 0; f < n_; ++f) {
                    const Aabb bx = bounds(id, int(f));
                    v[f] = bx.empty() ? -1 : (bx.lo.z - run_.groundZ() - m) / 0.2;
                }
                return v;
            });
            out.note = "airborne at " + (fmtTimes(run_, out.v).empty() ? std::string("no frame") : fmtTimes(run_, out.v));
            return out;
        }

        if (p == "turned" || p == "moved" || p == "slid_sideways") {
            std::string why;
            const auto f0 = anchor(a, why);
            if (!f0) return {out.v, why};
            const double want = number(a, 0, p == "turned" ? 30 : 0.5);
            double most = 0;
            out.v = perEntity(A, [&](const std::string& id) {
                std::vector<double> v(n_, -1.0);
                double total = 0;
                const auto p0 = run_.position(id, *f0);
                const auto fr0 = scene_.frameOf(id, *f0);
                double side = 0;
                for (size_t f = size_t(*f0); f < n_; ++f) {
                    double value = 0;
                    if (p == "turned") {
                        if (f > size_t(*f0)) total += std::remainder(heading(id, int(f)) - heading(id, int(f) - 1), 2 * kPi);
                        value = rad2deg(std::fabs(total));
                    } else if (const auto pf = run_.position(id, int(f)); pf && p0) {
                        const Vec3 d = *pf - *p0;
                        if (p == "moved") value = Vec3{d.x, d.y, d.z}.norm();
                        else {
                            const Vec3 y = fr0 ? fr0->q.axisY() : Vec3{0, 1, 0};
                            side = std::fabs(d.dot(Vec3{y.x, y.y, 0}.normalized()));
                            value = side;
                        }
                    }
                    most = std::max(most, value);
                    v[f] = (value - want) / std::max(p == "turned" ? 10.0 : 0.25, 0.25 * want);
                }
                return v;
            });
            out.note = std::string(p == "turned" ? "heading changed by up to " : p == "moved" ? "moved up to " : "slid sideways up to ") +
                       fmt(most) + (p == "turned" ? " deg" : " m") + " after t=" + fmt(run_.time(*f0)) + " s (needs " + fmt(want) + ")";
            return out;
        }

        if (p == "moving_away" || p == "approaching") {
            const double m = number(a, 0, 0.2);
            double best = -kInf;
            out.v = perEntity(A, [&](const std::string& x) {
                return perEntity(B, [&](const std::string& y) {
                    std::vector<double> v(n_, -1.0);
                    const auto vx = velocity(x), vy = velocity(y);
                    for (size_t f = 0; f < n_; ++f) {
                        const auto px = run_.position(x, int(f)), py = run_.position(y, int(f));
                        if (!px || !py) continue;
                        Vec3 d = *px - *py;
                        d.z = 0;
                        const double radial = (vx[f] - vy[f]).dot(d.normalized()) * (p == "moving_away" ? 1 : -1);
                        best = std::max(best, radial);
                        v[f] = (radial - m) / 0.3;
                    }
                    return v;
                });
            });
            out.note = std::string(p == "moving_away" ? "fastest moving away " : "fastest closing ") + fmt(best) + " m/s at " +
                       (fmtTimes(run_, out.v).empty() ? std::string("no frame above ") + fmt(m) : fmtTimes(run_, out.v));
            return out;
        }

        if (p == "in_front_of" || p == "behind" || p == "left_of" || p == "right_of" || p == "above" || p == "on" || p == "inside") {
            out.v = perEntity(A, [&](const std::string& x) {
                return perEntity(B, [&](const std::string& y) {
                    std::vector<double> v(n_, -1.0);
                    for (size_t f = 0; f < n_; ++f) {
                        const auto px = run_.position(x, int(f));
                        const auto fy = scene_.frameOf(y, int(f));
                        if (!px || !fy) continue;
                        const Vec3 l = fy->transformInv(*px);
                        const Vec3 half = scene_.facts().at(y).size * 0.5;
                        const Aabb ax = bounds(x, int(f)), by = bounds(y, int(f));
                        const double overlapX = std::min(ax.hi.x, by.hi.x) - std::max(ax.lo.x, by.lo.x);
                        const double overlapY = std::min(ax.hi.y, by.hi.y) - std::max(ax.lo.y, by.lo.y);
                        const double plan = std::min(overlapX, overlapY);
                        if (p == "in_front_of") v[f] = (l.x - half.x) / std::max(1.0, half.x);
                        else if (p == "behind") v[f] = (-l.x - half.x) / std::max(1.0, half.x);
                        else if (p == "left_of") v[f] = (l.y - half.y) / std::max(1.0, half.y);
                        else if (p == "right_of") v[f] = (-l.y - half.y) / std::max(1.0, half.y);
                        else if (p == "above") v[f] = std::min((ax.lo.z - by.hi.z + 0.05) / 0.2, plan / 0.2);
                        else if (p == "on") v[f] = std::min((0.08 - std::fabs(ax.lo.z - by.hi.z)) / 0.08, plan / 0.2);
                        else v[f] = by.contains(*px) ? 1 : -1;
                    }
                    return v;
                });
            });
            out.note = "holds at " + (fmtTimes(run_, out.v).empty() ? std::string("no frame") : fmtTimes(run_, out.v));
            return out;
        }

        if (p == "spins") {
            const double rpm = number(a, 0, 10);
            double most = 0;
            out.v = perEntity(A, [&](const std::string& id) {
                std::vector<double> v(n_, -1.0);
                for (size_t f = 1; f < n_; ++f) {
                    const auto f0 = scene_.frameOf(id, int(f - 1)), f1 = scene_.frameOf(id, int(f));
                    if (!f0 || !f1) continue;
                    const double r = (f0->q.conj() * f1->q).normalized().angle() * run_.fps() * 60 / (2 * kPi);
                    most = std::max(most, r);
                    v[f] = (r - rpm) / std::max(1.0, 0.2 * rpm);
                }
                return v;
            });
            out.note = "up to " + fmt(most) + " rpm";
            return out;
        }
        return {out.v, "predicate '" + p + "' is not grounded"};
    }

    // Per frame: permanent-or-not crush of a soft body, the 90th percentile of vertex displacement from rest (in its
    // host's frame, or with the median displacement removed for a free body).
    std::vector<double> crushOf(const std::string& id) const {
        std::vector<double> out(n_, 0);
        for (const PointSet& ps : run_.pointSets()) {
            if (ps.entity != id || ps.rest.empty()) continue;
            const auto hostParts = run_.partsOf(ps.host);
            for (size_t f = 0; f < n_ && f < ps.frames.size(); ++f) {
                const auto& pts = ps.frames[f];
                std::vector<double> d;
                for (size_t i = 0; i < pts.size() && i < ps.rest.size(); ++i)
                    d.push_back(((hostParts.empty() ? pts[i] : hostParts.front().get().poses[f].transformInv(pts[i])) - ps.rest[i]).norm());
                if (d.empty()) continue;
                std::sort(d.begin(), d.end());
                const double p90 = d[size_t(0.9 * double(d.size() - 1))];
                out[f] = hostParts.empty() ? p90 - d[d.size() / 2] : p90;
            }
        }
        return out;
    }

    // Particles leak from an object (or part): an emitter rides on it with its nozzle on (within 5 cm of) its
    // surface, and the first particles appear at its surface (within 15 cm), in the part's region if one is named.
    AtomValue leaks(const Formula& a, const std::string& fluid, const std::string& source) {
        AtomValue out;
        out.v.assign(n_, -1.0);
        const auto F = ents(fluid);
        if (F.empty()) return unbound(fluid);
        std::string host, region = "any";
        if (const auto pb = part(source)) host = pb->host, region = pb->region;
        else if (const auto S = ents(source); !S.empty()) host = S.front();
        if (host.empty()) return unbound(source);
        const Json* raw = scene_.raw(F.front());
        std::vector<std::string> problems;
        bool hosted = false, onSurface = false;
        const auto hostParts = run_.partsOf(host);
        if (raw)
            for (const Json& em : raw->value("emitters", Json::array())) {
                if (em.value("host", "") != host) continue;
                hosted = true;
                if (!em.contains("pos") || hostParts.empty()) continue;
                const Vec3 local{em["pos"][0].get<double>(), em["pos"][1].get<double>(), em["pos"][2].get<double>()};
                const Vec3 world = hostParts.front().get().poses[0].transform(local);
                double d = kInf;
                for (const Part& part : hostParts) d = std::min(d, part.sdf(0, world));
                for (const PointSet& ps : run_.pointSets())
                    if (ps.host == host && !ps.frames.empty())
                        for (const Vec3& q : ps.frames[0]) d = std::min(d, (q - world).norm());
                if (d <= 0.05 && inRegion(host, local, region)) onSurface = true;
                else
                    problems.push_back("its emitter at " + fmt(local.x) + ", " + fmt(local.y) + ", " + fmt(local.z) + " (in " + host +
                                       "'s frame) is " + fmt(d) + " m from " + host + "'s surface" +
                                       (inRegion(host, local, region) ? "" : " and not at its " + region) +
                                       ": put the nozzle on the surface of the " + (region == "any" ? host : source) +
                                       ", or the liquid appears in mid-air");
            }
        if (!hosted) problems.push_back(F.front() + " has no emitter with \"host\": \"" + host + "\", so it does not come from " + host);
        // Where the first particles appear.
        const PointSet* ps = nullptr;
        for (const PointSet& s : run_.pointSets()) if (s.entity == F.front()) ps = &s;
        int first = -1;
        if (ps)
            for (size_t f = 0; f < n_ && first < 0; ++f)
                if (!ps->frames[f].empty()) first = int(f);
        bool sourced = false;
        if (first >= 0 && !hostParts.empty()) {
            const Vec3 c = centroid(ps->frames[size_t(first)]);
            double d = kInf;
            for (const Part& part : hostParts) d = std::min(d, part.sdf(first, c));
            for (const PointSet& s : run_.pointSets())
                if (s.host == host && size_t(first) < s.frames.size())
                    for (const Vec3& q : s.frames[size_t(first)]) d = std::min(d, (q - c).norm());
            sourced = d <= 0.15;
            if (!sourced) problems.push_back("the first particles (t=" + fmt(run_.time(first)) + " s) appear " + fmt(d) + " m from " + host);
        } else if (first < 0) {
            problems.push_back(F.front() + " never has any particles");
        }
        const bool ok = hosted && onSurface && sourced;
        if (ok) std::fill(out.v.begin() + first, out.v.end(), 1.0);
        out.note = ok ? F.front() + " is emitted from " + host + "'s " + region + " surface, first at t=" + fmt(run_.time(first)) + " s" : "";
        for (size_t i = 0; i < problems.size(); ++i) out.note += (out.note.empty() ? "" : "; ") + problems[i];
        return out;
    }

    const RunData& run_;
    const Scene& scene_;
    const ParsedSpec& spec_;
    const Binding& b_;
    Visibility vis_;
    size_t n_;
    std::map<std::string, AtomValue> cache_;
};

// Camera binding: type, the object it rides on or is mounted on, its height, and words of the name in the id.
void bindCameras(const RunData& run, const ParsedSpec& spec, const Scene& scene, Binding& b, Report& r) {
    std::map<std::string, Json> camSpec;
    for (const Json& c : run.x().value("cameras", Json::array())) camSpec[c.value("id", "")] = c;
    std::set<std::string> used;
    for (const CameraSpec& cs : spec.cameras) {
        double best = -kInf;
        std::string pick;
        std::vector<std::string> pickProblems;
        for (const CameraTrack& cam : run.cameras()) {
            if (used.count(cam.id)) continue;
            const Json& j = camSpec[cam.id];
            const std::string type = j.value("type", "");
            std::vector<std::string> problems;
            double s = 0;
            if (cs.type != "any" && type != cs.type) s -= 3, problems.push_back("it is a " + type + " camera, the prompt's is " + cs.type);
            if (!cs.on.empty()) {
                const auto it = b.objects.find(cs.on);
                const bool rides = it != b.objects.end() && std::find(it->second.begin(), it->second.end(), cam.parent) != it->second.end();
                if (rides) s += 3;
                else s -= 3, problems.push_back("it rides on " + (cam.parent.empty() ? std::string("nothing") : cam.parent) + ", not on the " + cs.on);
            }
            if (!cs.mountedOn.empty()) {
                const auto it = b.objects.find(cs.mountedOn);
                double d = kInf;
                if (it != b.objects.end())
                    for (const auto& id : it->second)
                        for (const Part& p : run.partsOf(id)) d = std::min(d, p.sdf(0, cam.poses[0].p));
                if (d <= 1.5) s += 2;
                else s -= 2, problems.push_back("it is " + (d < kInf ? fmt(d) + " m" : std::string("far")) + " from the " + cs.mountedOn +
                                                " it should be mounted on (within 1.5 m)");
            }
            if (cs.minHeight > -kInf) {
                const double z = cam.poses[0].p.z - run.groundZ();
                if (z < cs.minHeight) s -= 1, problems.push_back("it is " + fmt(z) + " m above the ground (the prompt's is at least " + fmt(cs.minHeight) + " m)");
            }
            for (const auto& w : words(cam.id))
                for (const auto& v : words(cs.name + " " + cs.span))
                    if (w == v || (w.size() >= 3 && v.rfind(w, 0) == 0)) s += 1;
            if (s > best) best = s, pick = cam.id, pickProblems = problems;
        }
        if (pick.empty()) {
            r.add({"prompt_camera", Severity::Fail, cs.name, "the prompt's " + cs.name + " (" + cs.span + ") has no camera in the scene",
                   "add a " + (cs.type == "any" ? std::string("camera") : cs.type + " camera") + " for it"});
            continue;
        }
        used.insert(pick);
        b.cameras[cs.name] = pick;
        if (!pickProblems.empty()) {
            std::string m;
            for (const auto& p : pickProblems) m += (m.empty() ? "" : "; ") + p;
            r.add({"prompt_camera", Severity::Fail, cs.name, "camera " + pick + " stands for the prompt's " + cs.name + " (" + cs.span + "), but " + m,
                   "change camera " + pick + " to match the prompt"});
        } else {
            r.add({"prompt_camera", Severity::Pass, cs.name, "camera " + pick + " is the prompt's " + cs.name, ""});
        }
    }
    (void)scene;
}

}  // namespace

std::vector<std::string> lintPromptSpec(const Json& spec) {
    std::vector<std::string> errors;
    try {
        parseSpec(spec, errors);
    } catch (const std::exception& e) {
        errors.push_back(std::string("spec: ") + e.what());
    }
    return errors;
}

void runAdherence(const RunData& run, const Spec& sceneSpec, const Json& specJson, Report& r) {
    std::vector<std::string> errors;
    const ParsedSpec spec = parseSpec(specJson, errors);
    if (!errors.empty()) {
        std::string m = "the prompt spec is malformed:";
        for (const auto& e : errors) m += "\n  " + e;
        throw PromptSpecError(m);
    }
    std::string prompt;
    if (sceneSpec.has("meta")) prompt = sceneSpec["meta"].opaque().value("prompt", "");
    const auto squash = [](const std::string& s) {
        std::string o;
        for (char c : s) if (!std::isspace(static_cast<unsigned char>(c))) o += c;
        return o;
    };
    if (squash(prompt) != squash(spec.prompt))
        throw PromptSpecError("the prompt spec was written for another prompt than meta.prompt: write a spec for this prompt "
                              "with p2s_spec (spec prompt: \"" + spec.prompt.substr(0, 80) + "...\")");

    const Classes& cl = Classes::get();
    const Scene scene(run, sceneSpec);
    Binding b = bindObjects(spec, scene);

    // 1. Objects: present, count, class size, appearance.
    Table objects{"Prompt objects: what the prompt names and what stands for it",
                  "Bound from the prompt spec (written from the prompt alone, not from this scene). Size is length x width x height "
                  "at t = 0, including mounted soft bodies.",
                  {"prompt object", "class", "entity", "size m", "appearance", "verdict"}, {}};
    for (const ObjectSpec& o : spec.objects) {
        const auto& got = b.objects[o.name];
        const std::string desc = (o.color.empty() ? "" : o.color + " ") + o.cls + (o.count > 1 ? " x" + std::to_string(o.count) : "");
        if (got.empty()) {
            std::string kinds;
            for (const Json& k : cl.at(o.cls).value("kinds", Json::array())) kinds += (kinds.empty() ? "" : " or ") + k.get<std::string>();
            r.add({"prompt_object", Severity::Fail, o.name, "the prompt's " + o.name + " (" + (o.span.empty() ? desc : o.span) +
                   ") has no entity in the scene (a " + o.cls + " is a " + kinds + ")",
                   "add it as a " + kinds + " with \"appearance\": {\"class\": \"" + o.cls + "\"" + (o.color.empty() ? "" : ", \"color\": \"" + o.color + "\"") + "}"});
            objects.rows.push_back({o.name, desc, "-", "-", "-", "missing"});
            continue;
        }
        if (int(got.size()) < o.count)
            r.add({"prompt_object", Severity::Fail, o.name, "the prompt has " + std::to_string(o.count) + " " + o.name + ", the scene " +
                   std::to_string(got.size()) + " (" + got.front() + (got.size() > 1 ? ", ..." : "") + ")", "add the missing ones"});
        for (const std::string& id : got) {
            const EntityFacts& e = scene.facts().at(id);
            std::vector<std::string> verdicts;
            std::vector<std::string> sizeProblems;
            const double pen = sizePenalty(cl.at(o.cls).value("size_m", Json()), e.size, &sizeProblems);
            if (!sizeProblems.empty()) {
                std::string m;
                for (const auto& s : sizeProblems) m += (m.empty() ? "" : "; ") + s;
                const bool bad = pen > std::log(1.4);
                r.add({"prompt_object", bad ? Severity::Fail : Severity::Warn, id, id + " stands for the prompt's " + o.name + " (a " + o.cls +
                       ") but its " + m, "size it like a real " + o.cls});
                verdicts.push_back(bad ? "wrong size" : "size off");
            }
            if (!e.preset.empty() && cl.at(o.cls).contains("preset")) {
                const Json pr = cl.at(o.cls)["preset"];
                if (std::find(pr.begin(), pr.end(), e.preset) == pr.end()) {
                    r.add({"prompt_object", Severity::Fail, id, id + " stands for the prompt's " + o.name + " but uses the " + e.preset +
                           " preset", "use the " + pr[0].get<std::string>() + " preset"});
                    verdicts.push_back("wrong material");
                }
            }
            // Appearance: the class and colour the prompt gives must be stated, so the renderer and the video model
            // get them from the scene rather than from guesswork.
            std::string app = "-";
            const Json& ap = e.appearance;
            const bool needs = e.kind != "particles" && e.kind != "traffic_signal" && e.kind != "explosion";
            if (ap.is_object()) app = ap.dump();
            if (needs && (!ap.is_object() || !ap.contains("class"))) {
                r.add({"prompt_appearance", Severity::Fail, id, id + " (the prompt's " + o.name + ") has no appearance.class",
                       "add \"appearance\": {\"class\": \"" + o.cls + "\"" + (o.color.empty() ? "" : ", \"color\": \"" + o.color + "\"") + "} to " + id});
                verdicts.push_back("no class");
            } else if (needs) {
                const auto named = cl.classesNamed(ap["class"].get<std::string>());
                if (std::find(named.begin(), named.end(), o.cls) == named.end()) {
                    r.add({"prompt_appearance", Severity::Fail, id, id + " is a '" + ap["class"].get<std::string>() + "', but it stands for the prompt's " +
                           o.name + " (a " + o.cls + ")", "set appearance.class to \"" + o.cls + "\""});
                    verdicts.push_back("wrong class");
                }
            }
            if (!o.color.empty() && needs) {
                const std::string want = *cl.colorFamily(o.color);
                if (!ap.is_object() || !ap.contains("color")) {
                    r.add({"prompt_appearance", Severity::Fail, id, "the prompt's " + o.name + " is " + o.color + "; " + id + " has no appearance.color",
                           "add \"color\": \"" + o.color + "\" to " + id + "'s appearance"});
                    verdicts.push_back("no colour");
                } else if (const auto fam = familyOf(ap["color"]); !fam || !colorMatches(want, *fam)) {
                    r.add({"prompt_appearance", Severity::Fail, id, "the prompt's " + o.name + " is " + o.color + "; " + id + "'s colour " +
                           ap["color"].dump() + " reads as " + (fam ? *fam : std::string("an unknown colour word")), "set its colour to \"" + o.color + "\""});
                    verdicts.push_back("wrong colour");
                }
            }
            objects.rows.push_back({o.name, desc, id, fmt(e.size.x, 3) + " x " + fmt(e.size.y, 3) + " x " + fmt(e.size.z, 3), app,
                                    verdicts.empty() ? "ok" : [&] { std::string s; for (auto& v : verdicts) s += (s.empty() ? "" : ", ") + v; return s; }()});
        }
    }
    // Unprompted contacts with people: a pedestrian struck in a scene whose prompt never says so.
    {
        std::set<std::string> mentioned;  // pairs a requirement names in touches/hits/near
        for (const Requirement& q : spec.requirements)
            forEachAtom(*q.formula, [&](const Formula& a) {
                if (a.args.size() >= 2 && (a.pred == "touches" || a.pred == "hits"))
                    mentioned.insert(a.args[0].name + "|" + a.args[1].name), mentioned.insert(a.args[1].name + "|" + a.args[0].name);
            });
        std::map<std::string, std::string> objectOf;
        for (const auto& [name, ids] : b.objects) for (const auto& id : ids) objectOf[id] = name;
        for (const Json& c : run.x().value("contacts", Json::array())) {
            std::string x = c.value("a", ""), y = c.value("b", "");
            for (std::string* s : {&x, &y}) if (run.hasEntity(*s)) { const std::string h = run.entity(*s).desc.value("host", ""); if (!h.empty()) *s = h; }
            if (!objectOf.count(x) || !objectOf.count(y) || x == y) continue;
            // Matter that flows (liquids, sand, smoke) and cloth touch whatever holds or carries them.
            const auto flows = [&](const std::string& id) { const std::string k = scene.facts().at(id).kind; return k == "particles" || k == "cloth"; };
            if (flows(x) || flows(y)) continue;
            const std::string ox = objectOf[x], oy = objectOf[y];
            if (mentioned.count(ox + "|" + oy)) continue;
            const bool person = scene.facts().at(x).kind == "walker" || scene.facts().at(y).kind == "walker";
            r.add({"prompt_contact", person ? Severity::Fail : Severity::Warn, ox + "+" + oy,
                   "the " + ox + " (" + x + ") and the " + oy + " (" + y + ") touch at t=" + fmt(c.value("first_t", 0.0)) +
                       " s, which the prompt does not describe", person ? "keep them apart" : "keep them apart, or acknowledge it if the contact is incidental"});
        }
    }

    // 2. Parts: they must be where the prompt puts them; deformation is checked by requirements.
    Grounder g(run, scene, spec, b);
    for (const PartSpec& p : spec.parts) {
        const auto pb = g.part(p.name);
        if (!pb) continue;  // the object is missing: reported above
        std::string m = p.name + " is the " + p.region + " of " + pb->host;
        if (!pb->soft.empty()) {
            m += "; soft bodies there: ";
            for (size_t i = 0; i < pb->soft.size(); ++i) m += (i ? ", " : "") + pb->soft[i];
        }
        r.add({"prompt_part", Severity::Info, p.name, m, ""});
    }

    // 3. Cameras.
    bindCameras(run, spec, scene, b, r);

    // 4. Requirements.
    Table reqs{"Prompt requirements (temporal logic over the simulation)",
               "Each requirement of the prompt as a formula over measured predicates (spec/README.md). Robustness is in [-1, 1]: above 0 "
               "it holds, and its size says by how much.",
               {"id", "mode", "requirement", "formula", "robustness", "verdict"}, {}};
    std::map<std::string, std::pair<int, int>> byMode;
    int held = 0;
    for (const Requirement& q : spec.requirements) {
        const auto rho = robustness(*q.formula, run.frames(), run.fps(), [&](const Formula& a) { return g(a).v; });
        const double v = rho.empty() ? -1 : rho.front();
        const bool ok = v > 0;
        held += ok;
        auto& m = byMode[q.mode];
        m.first += ok, m.second += 1;
        std::string why;
        std::set<std::string> said;
        forEachAtom(*q.formula, [&](const Formula& a) {
            if (!said.insert(a.text()).second) return;
            const AtomValue& av = g(a);
            const std::string when = fmtTimes(run, av.v);
            why += (why.empty() ? "" : "; ") + a.text() + ": " + (when.empty() ? "never holds" : "holds " + when) +
                   (av.note.empty() ? "" : " (" + av.note + ")");
        });
        Json data = {{"robustness", v}};
        r.add({"prompt", ok ? Severity::Pass : q.implied ? Severity::Warn : Severity::Fail, q.id,
               std::string(q.implied ? "(implied by the prompt) " : "") + "\"" + q.text + "\" " + (ok ? "holds" : "does not hold") + ": " +
                   q.formula->text() + " has robustness " + fmt(v, 2) + ". " + why,
               ok ? "" : q.implied ? "make it happen, or acknowledge it if the prompt's scene really does otherwise"
                                   : "change the scene so that this happens as the prompt says (the measurements above say what is missing)",
               data});
        std::string cell = q.formula->text();  // a | inside a Markdown table cell ends the cell
        for (size_t at = cell.find('|'); at != std::string::npos; at = cell.find('|', at + 2)) cell.replace(at, 1, "\\|");
        reqs.rows.push_back({q.id, q.mode + (q.implied ? " (implied)" : ""), q.text, "`" + cell + "`", fmt(v, 2),
                             ok ? "holds" : q.implied ? "warns" : "FAILS"});
    }
    std::string summary;
    for (const auto& [mode, c] : byMode) summary += (summary.empty() ? "" : ", ") + mode + " " + std::to_string(c.first) + "/" + std::to_string(c.second);
    reqs.note += "\n\n" + std::to_string(held) + " of " + std::to_string(spec.requirements.size()) + " requirements hold (" + summary + ").";
    if (!spec.notSimulated.empty()) {
        reqs.note += " Left to the renderer and the video model (not checked here): ";
        for (size_t i = 0; i < spec.notSimulated.size(); ++i) reqs.note += (i ? "; " : "") + spec.notSimulated[i];
        reqs.note += ".";
    }
    r.table(objects);
    r.table(reqs);
    r.fact("prompt_adherence", {{"requirements", spec.requirements.size()}, {"hold", held}});
}

}  // namespace p2s
