#include "core/Camera.h"

#include <random>

namespace p2s {

Quat lookRotation(const Vec3& eye, const Vec3& target) {
    const Vec3 f = (target - eye).normalized();
    Vec3 r = f.cross({0, 0, 1});
    if (r.norm2() < 1e-12) r = f.cross({0, 1, 0});  // looking straight up or down
    r = r.normalized();
    const Vec3 u = r.cross(f);
    return Quat::fromAxes(r, u, -f);
}

Camera::Camera(const Spec& s) : spec_(s.raw()) {
    id_ = s.str("id");
    type_ = s.oneOf("type", {"fixed", "attached", "handheld", "orbit", "follow"});
    intr_.fovDeg = s.positive("fov_deg", 60);
    if (intr_.fovDeg >= 170) s["fov_deg"].fail("must be < 170");
    if (s.has("resolution")) {
        const auto r = s.numbers("resolution");
        if (r.size() != 2 || r[0] < 16 || r[1] < 16) s["resolution"].fail("expected [width, height] in pixels");
        intr_.width = int(r[0]);
        intr_.height = int(r[1]);
    }
    if (s.has("clip_m")) {
        const auto c = s.numbers("clip_m");
        if (c.size() != 2 || c[0] <= 0 || c[1] <= c[0]) s["clip_m"].fail("expected [near, far] with 0 < near < far");
        intr_.nearM = c[0];
        intr_.farM = c[1];
    }
    if (type_ == "fixed" || type_ == "handheld") {
        if (type_ == "handheld" && s.has("parent")) parent_ = s.str("parent");
        else position_ = s.vec3("position");
        if (s.has("look_at_entity")) lookEntity_ = s.str("look_at_entity");
        else lookAt_ = s.vec3("look_at");
    }
    if (type_ == "attached") parent_ = s.str("parent");
    if (s.has("offset") && type_ != "follow") {
        const Spec o = s["offset"];
        offsetPos_ = o.vec3("pos", {0, 0, 0});
        offsetRot_ = o.orientation();
    }
    if (type_ == "handheld" && s.has("shake")) {
        const Spec sh = s["shake"];
        shakeRot_ = deg2rad(sh.num("rot_deg", 0.6));
        shakePos_ = sh.num("pos_m", 0.004);
        shakeSeed_ = unsigned(sh.integer("seed", 11));
    }
    if (type_ == "orbit") {
        if (s.has("center_entity")) centerEntity_ = s.str("center_entity");
        else center_ = s.vec3("center");
        radius_ = s.positive("radius_m");
        height_ = s.num("height_m");
        degPerS_ = s.num("deg_per_s", 10);
        startDeg_ = s.num("start_deg", 0);
    }
    if (type_ == "follow") {
        parent_ = s.str("target");
        offsetPos_ = s.vec3("offset");
        lagS_ = s.num("lag_s", 0.3);
    }
}

std::vector<std::string> Camera::referencedEntities() const {
    std::vector<std::string> out;
    for (const auto& e : {parent_, lookEntity_, centerEntity_})
        if (!e.empty()) out.push_back(e);
    return out;
}

Vec3 Camera::target(const PoseOf& poseOf) const { return lookEntity_.empty() ? lookAt_ : poseOf(lookEntity_).p; }

Pose Camera::shake(double t) const {
    // A few incommensurate sines per axis, fixed by the seed: a hand holding a phone.
    std::mt19937 rng(shakeSeed_);
    std::uniform_real_distribution<double> f(1.0, 3.0), ph(0, 2 * kPi);
    double r[3] = {0, 0, 0}, p[3] = {0, 0, 0};
    for (int k = 0; k < 3; ++k)
        for (int i = 0; i < 3; ++i) {
            r[k] += std::sin(2 * kPi * f(rng) * t + ph(rng)) / 3.0;
            p[k] += std::sin(2 * kPi * f(rng) * t + ph(rng)) / 3.0;
        }
    return {Vec3{p[0], p[1], p[2]} * shakePos_, Quat::fromYawPitchRoll(r[2] * shakeRot_, r[1] * shakeRot_, r[0] * shakeRot_)};
}

Pose Camera::update(double t, const PoseOf& poseOf) {
    Pose out;
    if (type_ == "fixed") {
        out = {position_, lookRotation(position_, target(poseOf))};
    } else if (type_ == "attached") {
        // Parent +X is forward and +Z up; the camera's -Z must point along parent +X.
        const Quat base = Quat::fromAxes({0, -1, 0}, {0, 0, 1}, {-1, 0, 0});
        const Pose par = poseOf(parent_);
        out = {par.transform(offsetPos_), par.q * offsetRot_ * base};
    } else if (type_ == "handheld") {
        const Vec3 eye = parent_.empty() ? position_ : poseOf(parent_).transform(offsetPos_);
        const Pose base{eye, lookRotation(eye, target(poseOf))};
        out = base * shake(t);
    } else if (type_ == "orbit") {
        const Vec3 c = centerEntity_.empty() ? center_ : poseOf(centerEntity_).p;
        const double a = deg2rad(startDeg_ + degPerS_ * t);
        const Vec3 eye = c + Vec3{radius_ * std::cos(a), radius_ * std::sin(a), height_};
        out = {eye, lookRotation(eye, c)};
    } else {  // follow: chase the target with a first-order lag
        const Vec3 tgt = poseOf(parent_).p;
        const Vec3 want = tgt + offsetPos_;
        if (!followPos_) followPos_ = want;
        const double alpha = lagS_ > 0 ? 1.0 - std::exp(-(t - lastT_) / lagS_) : 1.0;
        followPos_ = *followPos_ + (want - *followPos_) * alpha;
        out = {*followPos_, lookRotation(*followPos_, tgt)};
    }
    lastT_ = t;
    return out;
}

Json Camera::toJson() const {
    return {{"id", id_},
            {"type", type_},
            {"fov_deg", intr_.fovDeg},
            {"resolution", {intr_.width, intr_.height}},
            {"clip_m", {intr_.nearM, intr_.farM}},
            {"spec", spec_}};
}

Projection project(const Pose& cam, const Intrinsics& k, const Vec3& world) {
    const Vec3 c = cam.transformInv(world);  // camera frame: x right, y up, looking down -z
    const double depth = -c.z;
    const double f = k.focalPx();
    if (depth <= 1e-9) return {-1, -1, depth};
    return {k.width / 2.0 + f * c.x / depth, k.height / 2.0 - f * c.y / depth, depth};
}

}  // namespace p2s
