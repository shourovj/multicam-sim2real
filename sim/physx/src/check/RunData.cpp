#include "check/RunData.h"

#include <fstream>

namespace p2s {

namespace {

std::vector<Pose> poseRows(const std::vector<float>& data, int frames, int rows, int row) {
    std::vector<Pose> out;
    for (int f = 0; f < frames; ++f) {
        const float* p = &data[size_t((f * rows + row) * 7)];
        out.push_back({{p[0], p[1], p[2]}, Quat{p[3], p[4], p[5], p[6]}.normalized()});
    }
    return out;
}

}  // namespace

RunData::RunData(const std::string& dir) : dir_(dir) {
    std::ifstream in(dir + "/x.json");
    if (!in) throw SpecError("no x.json in " + dir + " (run p2s_sim first)");
    x_ = Json::parse(in);
    frames_ = x_.at("clock").at("frames").get<int>();
    fps_ = x_.at("clock").at("fps").get<double>();
    const StreamReader sr(dir, x_.at("streams"));
    const Json assemblies = x_.value("assemblies", Json::object());

    for (const Json& e : x_.at("entities")) {
        EntityInfo info;
        info.id = e.at("id").get<std::string>();
        info.kind = e.at("kind").get<std::string>();
        info.desc = e;
        info.assembly = assemblies.value(info.id, "");
        info.mass = e.value("mass_kg", 0.0);
        const std::string velKey = "body/" + info.id + "/vel";
        if (sr.has(velKey)) {
            const auto v = sr.read(velKey);
            for (int f = 0; f < frames_; ++f) info.velocity.push_back({v[size_t(f * 6)], v[size_t(f * 6 + 1)], v[size_t(f * 6 + 2)]});
        }
        if (info.id == "ground") {
            hasGround_ = true;
            const Json& slab = e.at("parts")[0].at("shapes")[0];
            groundZ_ = slab.at("pos")[2].get<double>() + slab.at("half_extents")[2].get<double>();
            const auto gp = sr.read("body/ground/pose");
            groundZ_ += gp[2];
        }
        if (e.contains("parts")) loadParts(e, info, sr);
        if (info.kind == "soft_body" || info.kind == "cloth" || info.kind == "particles") loadPoints(e, sr);
        if (info.kind == "joint") exempt_.insert(std::minmax(e.at("a").get<std::string>(), e.at("b").get<std::string>()));
        entities_[info.id] = info;
    }
    for (const auto& [a, ia] : entities_)
        for (const auto& [b, ib] : entities_)
            if (a < b && !ia.assembly.empty() && ia.assembly == ib.assembly) exempt_.insert({a, b});
    for (const PointSet& p : points_)
        if (!p.host.empty()) exempt_.insert(std::minmax(p.entity, p.host));

    for (const Json& c : x_.at("cameras")) {
        CameraTrack cam;
        cam.id = c.at("id").get<std::string>();
        const Json& spec = c.value("spec", Json::object());
        cam.parent = spec.value("parent", spec.value("target", ""));
        if (spec.value("type", "") == "follow") cam.parent.clear();  // a chase camera is outside its target
        cam.k.fovDeg = c.at("fov_deg").get<double>();
        cam.k.width = c.at("resolution")[0].get<int>();
        cam.k.height = c.at("resolution")[1].get<int>();
        cam.k.nearM = c.at("clip_m")[0].get<double>();
        cam.k.farM = c.at("clip_m")[1].get<double>();
        cam.poses = poseRows(sr.read("camera/" + cam.id), frames_, 1, 0);
        cameras_.push_back(cam);
    }
}

void RunData::loadParts(const Json& e, const EntityInfo& info, const StreamReader& sr) {
    for (const Json& pj : e.at("parts")) {
        Part p;
        p.entity = info.id;
        p.name = pj.at("name").get<std::string>();
        p.kind = pj.value("kind", info.kind);
        for (const Json& s : pj.at("shapes")) p.shapes.push_back(Shape::fromJson(s));
        const std::string key = pj.at("stream").get<std::string>();
        const auto data = sr.read(key);
        const int rows = int(data.size() / size_t(7 * frames_));
        p.poses = poseRows(data, frames_, rows, pj.at("row").get<int>());
        for (const Shape& s : p.shapes) {
            for (const Vec3& v : s.surfaceSamples()) p.samples.push_back(s.local.transform(v));
            const Aabb b = s.bounds().transformed(s.local);
            p.local.add(b.lo);
            p.local.add(b.hi);
            const Vec3 ext = s.bounds().extent() * 0.5;
            p.minHalfExtent = std::min(p.minHalfExtent, std::min({ext.x, ext.y, ext.z}));
        }
        parts_.push_back(std::move(p));
    }
}

void RunData::loadPoints(const Json& e, const StreamReader& sr) {
    PointSet ps;
    ps.entity = e.at("id").get<std::string>();
    ps.kind = e.at("kind").get<std::string>();
    ps.host = e.value("host", "");
    const Json& streams = e.at("streams");
    const std::string key = streams.contains("verts") ? streams.at("verts").get<std::string>()
                                                     : streams.at("pos").get<std::string>();
    const auto data = sr.read(key);
    if (x_.at("streams").at(key).at("rows") == "variable") {
        const auto counts = sr.counts(key);
        size_t off = 0;
        for (int32_t c : counts) {
            std::vector<Vec3> f;
            for (int32_t i = 0; i < c; ++i, off += 3) f.push_back({data[off], data[off + 1], data[off + 2]});
            ps.frames.push_back(std::move(f));
        }
        ps.radius = e.value("particle_radius_m", 0.005);
        ps.particleKind = e.value("particle_kind", "fluid");
    } else {
        const size_t n = data.size() / size_t(3 * frames_);
        for (int f = 0; f < frames_; ++f) {
            std::vector<Vec3> fr;
            for (size_t i = 0; i < n; ++i) {
                const size_t o = (size_t(f) * n + i) * 3;
                fr.push_back({data[o], data[o + 1], data[o + 2]});
            }
            ps.frames.push_back(std::move(fr));
        }
    }
    if (streams.contains("rest")) {
        const auto r = sr.read(streams.at("rest").get<std::string>());
        for (size_t i = 0; i + 2 < r.size(); i += 3) ps.rest.push_back({r[i], r[i + 1], r[i + 2]});
    }
    if (streams.contains("tets")) {
        const auto t = sr.readInts(streams.at("tets").get<std::string>());
        for (size_t i = 0; i + 3 < t.size(); i += 4) ps.tets.push_back({t[i], t[i + 1], t[i + 2], t[i + 3]});
    }
    points_.push_back(std::move(ps));
}

int RunData::frameAt(double t) const { return std::clamp(int(std::lround(t * fps_)), 0, frames_ - 1); }

const EntityInfo& RunData::entity(const std::string& id) const {
    const auto it = entities_.find(id);
    if (it == entities_.end()) throw SpecError("no entity '" + id + "' in the simulation output");
    return it->second;
}

bool RunData::exempt(const std::string& a, const std::string& b) const {
    return a == b || exempt_.count(std::minmax(a, b)) > 0;
}

std::optional<Vec3> RunData::position(const std::string& id, int frame) const {
    for (const Part& p : parts_)
        if (p.entity == id) return p.poses[size_t(frame)].p;
    for (const PointSet& p : points_)
        if (p.entity == id && !p.frames[size_t(frame)].empty()) {
            Vec3 c;
            for (const Vec3& v : p.frames[size_t(frame)]) c += v;
            return c / double(p.frames[size_t(frame)].size());
        }
    return std::nullopt;
}

std::vector<Vec3> RunData::samples(const std::string& id, int frame, size_t maxPoints) const {
    std::vector<Vec3> out;
    for (const Part& p : parts_)
        if (p.entity == id)
            for (const Vec3& s : p.samples) out.push_back(p.poses[size_t(frame)].transform(s));
    for (const PointSet& p : points_)
        if (p.entity == id) {
            const auto& f = p.frames[size_t(frame)];
            const size_t stride = std::max<size_t>(1, f.size() / maxPoints);
            for (size_t i = 0; i < f.size(); i += stride) out.push_back(f[i]);
        }
    if (out.size() > maxPoints) {
        std::vector<Vec3> thin;
        const size_t stride = out.size() / maxPoints + 1;
        for (size_t i = 0; i < out.size(); i += stride) thin.push_back(out[i]);
        out = std::move(thin);
    }
    return out;
}

std::optional<double> RunData::eventTime(const std::string& name) const {
    for (const Json& e : x_.at("events"))
        if (e.at("name").get<std::string>() == name) return e.at("t").get<double>();
    return std::nullopt;
}

std::vector<std::array<double, 4>> RunData::energy() const {
    const auto data = StreamReader(dir_, x_.at("streams")).read("world/energy");
    std::vector<std::array<double, 4>> out;
    for (size_t i = 0; i + 3 < data.size(); i += 4) out.push_back({data[i], data[i + 1], data[i + 2], data[i + 3]});
    return out;
}

}  // namespace p2s
