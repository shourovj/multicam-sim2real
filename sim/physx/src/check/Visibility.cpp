#include "check/Visibility.h"

#include <algorithm>
#include <cmath>

namespace p2s {

namespace {

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

}  // namespace

const View& Visibility::at(const CameraTrack& cam, const std::string& entity, int f) {
    auto& track = cache_[{cam.id, entity}];
    if (track.empty()) track.resize(size_t(run_.frames()));
    auto& slot = track[size_t(f)];
    if (!slot) slot = compute(cam, entity, f);
    return *slot;
}

const std::vector<Aabb>& Visibility::bounds(int f) {
    auto& b = bounds_[size_t(f)];
    if (b.empty())
        for (const Part& p : run_.parts()) b.push_back(p.worldBounds(f));
    return b;
}

View Visibility::compute(const CameraTrack& cam, const std::string& entity, int f) {
    View v;
    const Pose& cp = cam.poses[size_t(f)];
    const auto pts = run_.samples(entity, f, 160);
    if (pts.empty()) return v;
    std::vector<std::pair<Vec3, Projection>> framed;
    Aabb reach;  // spans the camera and every framed point: only parts meeting it can occlude
    reach.add(cp.p);
    for (const Vec3& p : pts)
        if (const Projection pr = project(cp, cam.k, p); inClip(pr, cam.k)) {
            framed.push_back({p, pr});
            reach.add(p);
        }
    const auto& parts = run_.parts();
    const auto& wb = bounds(f);
    std::vector<size_t> occluders;
    for (size_t i = 0; i < parts.size(); ++i)
        if (parts[i].entity != entity && parts[i].entity != cam.parent && !run_.exempt(parts[i].entity, entity) &&
            overlaps(wb[i], reach))
            occluders.push_back(i);
    Aabb screen;
    int vis = 0, facing = 0;
    const auto own = run_.partsOf(entity);
    for (const auto& [p, pr] : framed) {
        const double dist = (p - cp.p).norm();
        const Vec3 d = (p - cp.p) / dist;  // dist > near clip > 0
        // Points on the far side of the entity itself (a plate's underside) are not counted either way.
        const bool backside = std::any_of(own.begin(), own.end(), [&](const Part& o) {
            const auto hit = PosedShapes{o.shapes, o.poses[size_t(f)]}.raycast(cp.p, d, dist);
            return hit && *hit < dist - 0.05;
        });
        if (backside) continue;
        ++facing;
        const bool hidden = std::any_of(occluders.begin(), occluders.end(), [&](size_t i) {
            if (!rayHitsBox(cp.p, d, dist, wb[i])) return false;
            const auto hit = PosedShapes{parts[i].shapes, parts[i].poses[size_t(f)]}.raycast(cp.p, d, dist);
            return hit && *hit < dist - 0.05;
        });
        if (hidden) continue;
        ++vis;
        screen.add({pr.u, pr.v, 0});
    }
    v.inFrame = double(framed.size()) / double(pts.size());
    v.visible = facing > 0 ? v.inFrame * double(vis) / double(facing) : 0.0;  // share of what faces the camera
    if (vis > 0) {
        v.widthPx = screen.extent().x;
        v.heightPx = screen.extent().y;
        v.area = (v.widthPx * v.heightPx) / double(cam.k.width * cam.k.height);
    }
    return v;
}

}  // namespace p2s
