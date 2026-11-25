#include "bodies/Walker.h"

using namespace physx;

namespace p2s {

namespace {
constexpr double kStrideM = 1.5;  // one gait cycle; phase advances with distance so the feet keep pace
}

Walker::Walker(const Spec& s, World& w) : Entity(s.str("id"), "walker") {
    if (s.has("path")) {
        path_ = Polyline::parse(s["path"]);
        speed_ = s.positive("speed_mps", 1.4);
        startS_ = s.num("start_s", 0);
        offset_ = s.num("start_offset_m", 0);
        if (offset_ < 0) s["start_offset_m"].fail("must be >= 0");
    } else {
        const auto at = s.numbers("idle_at");
        if (at.size() != 2) s["idle_at"].fail("expected [x, y]");
        idleAt_ = {at[0], at[1], 0};
        yaw0_ = deg2rad(s.num("facing_deg", 0));
        mode_ = s.oneOf("pose", {"stand", "phone"}, "stand");
        sway_ = s.num("sway_m", 0.02);
    }
    height_ = s.positive("height_m", 1.75);
    radius_ = s.positive("radius_m", 0.25);
    groundZ_ = s.num("ground_z", 0);
    if (height_ < 2 * radius_) s["height_m"].fail("must be at least twice radius_m");
    shapes_ = {Shape::capsule(radius_, height_ / 2 - radius_, Pose{{0, 0, height_ / 2}, {}})};
    shapes_.front().material = "flesh";
    actor_.reset(w.physics().createRigidDynamic(px(poseAt(0))));
    w.shapeFactory().attach(*actor_, shapes_, w.filterFor(id()), id());
    actor_->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
    PxRigidBodyExt::setMassAndUpdateInertia(*actor_, 75.0f);
    w.scene().addActor(*actor_);
}

double Walker::distance(double t) const {
    return std::min(offset_ + speed_ * std::max(0.0, t - startS_), path_.length());
}

bool Walker::walking(double t) const {
    return mode_ == "walk" && t >= startS_ && distance(t) < path_.length() - 1e-6;
}

Pose Walker::poseAt(double t) const {
    if (mode_ == "walk") {
        const double d = distance(t);
        const Vec3 dir = path_.tangent(d);
        return {path_.at(d) + Vec3{0, 0, groundZ_}, Quat::axisAngle({0, 0, 1}, std::atan2(dir.y, dir.x))};
    }
    // Standing people are never quite still.
    const int k = int(id().size());
    const Vec3 sway{sway_ * std::sin(2 * kPi * 0.31 * t + k), sway_ * std::sin(2 * kPi * 0.19 * t + 2 * k), groundZ_};
    return {idleAt_ + sway, Quat::axisAngle({0, 0, 1}, yaw0_ + 0.04 * std::sin(2 * kPi * 0.23 * t + k))};
}

Vec3 Walker::velocity() const {
    if (!walking(t_)) return {};
    return poseAt(t_).q.axisX() * speed_;
}

void Walker::beforeStep(World& w, double dt) {
    t_ = w.time();
    actor_->setKinematicTarget(px(poseAt(w.time() + dt)));
}

void Walker::afterStep(World& w, double) { t_ = w.time(); }  // pose() matches the actor for post-step queries

void Walker::record(World& w, Recorder& rec) {
    t_ = w.time();
    putPose(rec, "body/" + id() + "/pose", pose());
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
