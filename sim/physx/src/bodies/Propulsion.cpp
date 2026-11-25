#include "bodies/Propulsion.h"

#include "bodies/Joint.h"

using namespace physx;

namespace p2s {

Buoyancy::Buoyancy(const std::vector<Shape>& shapes, double dragCd) : cd_(dragCd) {
    // A 6x6x6 grid over each shape's bounds; cells inside the shape share its volume.
    constexpr int n = 6;
    for (const Shape& s : shapes) {
        const Aabb b = s.bounds();
        const Vec3 step = b.extent() / n;
        std::vector<Vec3> inside;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < n; ++k) {
                    const Vec3 p = b.lo + Vec3{(i + 0.5) * step.x, (j + 0.5) * step.y, (k + 0.5) * step.z};
                    if (s.sdf(p) < 0) inside.push_back(p);
                }
        if (inside.empty()) inside.push_back(b.center());
        const double v = s.volume() / double(inside.size());
        for (const Vec3& p : inside) cells_.push_back({s.local.transform(p), v, std::pow(v, 2.0 / 3.0) / n});  // areas sum to about V^(2/3)
    }
}

Buoyancy::Result Buoyancy::apply(PxRigidDynamic& body, const Water& water, const Vec3& gravity) const {
    const Pose pose = fromPx(body.getGlobalPose());
    const Vec3 com = fromPx((body.getGlobalPose() * body.getCMassLocalPose()).p);
    const Vec3 v = fromPx(body.getLinearVelocity()), om = fromPx(body.getAngularVelocity());
    double wet = 0, total = 0, power = 0;
    for (const Cell& c : cells_) {
        total += c.volume;
        const Vec3 p = pose.transform(c.local);
        if (!water.wet(p)) continue;
        wet += c.volume;
        const Vec3 vp = v + om.cross(p - com), rel = vp - water.current;
        const Vec3 f = gravity * (-water.density * c.volume) - rel * (0.5 * water.density * cd_ * c.area * rel.norm());
        PxRigidBodyExt::addForceAtPos(body, px(f), px(p));
        power += f.dot(vp);
    }
    return {total > 0 ? wet / total : 0.0, power};
}

Thruster::Thruster(const Spec& s) {
    id_ = s.str("id");
    pos_ = s.vec3("pos", {0, 0, 0});
    dir_ = s.vec3("direction").normalized();
    if (dir_.norm2() < 0.5) s["direction"].fail("must be a non-zero vector");
    if (s.has("joint")) {
        jointId_ = s.str("joint");
        k_ = s.positive("k");
    } else {
        force_ = s.positive("force_n");
    }
    if (s.has("burn_s")) {
        const auto b = s.numbers("burn_s");
        if (b.size() != 2 || b[1] <= b[0]) s["burn_s"].fail("expected [start_s, end_s]");
        burn_ = std::array<double, 2>{b[0], b[1]};
    }
    active_ = s.flag("active", true);
    medium_ = s.oneOf("medium", {"any", "water", "air"}, "any");
}

void Thruster::link(World& w, const std::string& owner, double ownerMass) {
    if (!jointId_.empty()) {
        Joint& j = w.entityAs<Joint>(jointId_, owner + ".thrusters." + id_);
        if (!j.isRevolute()) throw SpecError(owner + ".thrusters." + id_ + ": joint '" + jointId_ + "' must be revolute");
        joint_ = j;
    }
    actuatorId_ = owner + "/" + id_;
    w.actuators().declare(actuatorId_, "thruster", ownerMass);
}

void Thruster::apply(PxRigidDynamic& body, World& w, double dt) {
    last_ = 0;
    const Pose pose = fromPx(body.getGlobalPose());
    const Vec3 at = pose.transform(pos_);
    const bool burning = active_ && (!burn_ || (w.time() >= (*burn_)[0] && w.time() < (*burn_)[1]));
    const bool wet = w.water() && w.water()->wet(at);
    const bool bites = medium_ == "any" || (medium_ == "water") == wet;
    if (burning && bites) {
        if (joint_) {
            const double omega = joint_->get().speed();
            last_ = k_ * omega * std::fabs(omega);
        } else {
            last_ = force_;
        }
    }
    const Vec3 f = pose.q.rotate(dir_) * last_;
    if (last_ != 0) PxRigidBodyExt::addForceAtPos(body, px(f), px(at));
    w.actuators().step(actuatorId_, last_, f.dot(fromPx(PxRigidBodyExt::getVelocityAtPos(body, px(at)))), dt);
}

Json Thruster::toJson() const {
    Json j = {{"id", id_}, {"pos", p2s::toJson(pos_)}, {"direction", p2s::toJson(dir_)}, {"medium", medium_},
              {"active", active_}};
    if (burn_) j["burn_s"] = {(*burn_)[0], (*burn_)[1]};
    if (jointId_.empty()) j["force_n"] = force_;
    else j["joint"] = jointId_, j["k"] = k_;
    return j;
}

}  // namespace p2s
