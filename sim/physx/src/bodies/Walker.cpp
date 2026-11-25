#include "bodies/Walker.h"

using namespace physx;

namespace p2s {

namespace {
constexpr double kStrideM = 1.5;  // one gait cycle; phase advances with distance so the feet keep pace
}

Walker::Walker(const Spec& s, World& w) : Entity(s.str("id"), "walker") {
    if (s.has("path")) {
        const Spec p = s["path"];
        for (size_t i = 0; i < p.size(); ++i) {
            const auto& xy = p.at(i).raw();
            if (!xy.is_array() || xy.size() != 2) p.at(i).fail("expected [x, y]");
            path_.push_back({xy[0].get<double>(), xy[1].get<double>()});
        }
        if (path_.size() < 2) p.fail("needs at least two points");
        knots_ = {0};
        for (size_t i = 1; i < path_.size(); ++i)
            knots_.push_back(knots_.back() + std::hypot(path_[i][0] - path_[i - 1][0], path_[i][1] - path_[i - 1][1]));
        speed_ = s.positive("speed_mps", 1.4);
        startS_ = s.num("start_s", 0);
        offset_ = s.num("start_offset_m", 0);
    } else {
        const Vec3 at{s["idle_at"].at(0).asNum(), s["idle_at"].at(1).asNum(), 0};
        path_ = {{at.x, at.y}};
        yaw0_ = deg2rad(s.num("facing_deg", 0));
        mode_ = s.oneOf("pose", {"stand", "phone"}, "stand");
        sway_ = s.num("sway_m", 0.02);
    }
    height_ = s.positive("height_m", 1.75);
    radius_ = s.positive("radius_m", 0.25);
    groundZ_ = s.num("ground_z", 0);
    shapes_ = {Shape::capsule(radius_, height_ / 2 - radius_, Pose{{0, 0, height_ / 2}, {}})};
    shapes_.front().material = "flesh";
    actor_.reset(w.physics().createRigidDynamic(px(poseAt(0))));
    w.shapeFactory().attach(*actor_, shapes_, w.filterFor(id()), id());
    actor_->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
    PxRigidBodyExt::setMassAndUpdateInertia(*actor_, 75.0f);
    w.scene().addActor(*actor_);
}

double Walker::distance(double t) const {
    return std::min(offset_ + speed_ * std::max(0.0, t - startS_), knots_.back());
}

bool Walker::walking(double t) const {
    return mode_ == "walk" && t >= startS_ && distance(t) < knots_.back() - 1e-6;
}

Pose Walker::poseAt(double t) const {
    double x = path_[0][0], y = path_[0][1], yaw = yaw0_;
    if (mode_ == "walk") {
        const double d = distance(t);
        size_t i = 0;
        while (i + 2 < knots_.size() && knots_[i + 1] < d) ++i;
        const double u = (d - knots_[i]) / std::max(knots_[i + 1] - knots_[i], 1e-9);
        x = path_[i][0] + u * (path_[i + 1][0] - path_[i][0]);
        y = path_[i][1] + u * (path_[i + 1][1] - path_[i][1]);
        yaw = std::atan2(path_[i + 1][1] - path_[i][1], path_[i + 1][0] - path_[i][0]);
    } else {  // standing people are never quite still
        const int k = int(id().size());
        x += sway_ * std::sin(2 * kPi * 0.31 * t + k);
        y += sway_ * std::sin(2 * kPi * 0.19 * t + 2 * k);
        yaw += 0.04 * std::sin(2 * kPi * 0.23 * t + k);
    }
    return {{x, y, groundZ_}, Quat::axisAngle({0, 0, 1}, yaw)};
}

Vec3 Walker::velocity() const {
    if (!walking(t_)) return {};
    return poseAt(t_).q.axisX() * speed_;
}

void Walker::beforeStep(World& w, double dt) {
    t_ = w.time();
    actor_->setKinematicTarget(px(poseAt(w.time() + dt)));
}

void Walker::record(World& w, Recorder& rec) {
    t_ = w.time();
    const Pose p = pose();
    rec.put("body/" + id() + "/pose",
            {float(p.p.x), float(p.p.y), float(p.p.z), float(p.q.x), float(p.q.y), float(p.q.z), float(p.q.w)});
    const bool walk = walking(t_);
    const float phase = mode_ == "walk" ? float(std::fmod(distance(t_) / kStrideM, 1.0)) : 0.0f;
    const float poseId = mode_ == "walk" ? (walk ? 0.0f : 1.0f) : (mode_ == "phone" ? 2.0f : 1.0f);
    rec.put("walker/" + id(), {phase, walk ? 1.0f : 0.0f, walk ? float(speed_) : 0.0f, poseId});
}

Json Walker::describe() const {
    Json shapes = Json::array();
    for (const Shape& s : shapes_) shapes.push_back(s.toJson());
    return {{"parts", {{{"name", "body"}, {"shapes", shapes}, {"stream", "body/" + id() + "/pose"}, {"row", 0}}}},
            {"height_m", height_},
            {"mode", mode_},
            {"streams", {{"pose", "body/" + id() + "/pose"}, {"state", "walker/" + id()}}},
            {"stream_columns", {"walk_phase", "walking", "speed_mps", "pose_id (0 walk, 1 stand, 2 phone)"}}};
}

}  // namespace p2s
