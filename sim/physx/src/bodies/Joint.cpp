#include "bodies/Joint.h"

using namespace physx;

namespace p2s {

Joint::Joint(const Spec& s, World&) : Entity(s.str("id"), "joint") {
    type_ = s.oneOf("type", {"fixed", "revolute", "prismatic", "spherical", "distance"});
    a_ = s.str("a", "world");
    b_ = s.str("b");
    if (a_ == b_) s["b"].fail("a joint needs two different bodies");
    anchor_ = s.vec3("anchor");
    if (type_ == "revolute" || type_ == "prismatic") {
        axis_ = s.vec3("axis").normalized();
        if (axis_.norm2() < 0.5) s["axis"].fail("must be a non-zero vector");
    }
    if (s.has("limits")) {
        if (type_ != "revolute" && type_ != "prismatic") s["limits"].fail("only revolute and prismatic joints have limits");
        limits_ = Range{s["limits"].num("lower"), s["limits"].num("upper")};
        if (limits_->upper <= limits_->lower) s["limits"].fail("upper must be > lower");
    }
    if (s.has("motor")) {
        if (type_ != "revolute") s["motor"].fail("only revolute joints have a motor");
        motor_ = Motor{s["motor"].num("velocity"), s["motor"].positive("max_force")};
    }
    if (type_ == "distance") {
        if (s.has("spring")) spring_ = Spring{s["spring"].positive("stiffness"), s["spring"].num("damping", 0)};
        if (s.has("min_distance")) minDistance_ = s.num("min_distance");
        if (s.has("max_distance")) maxDistance_ = s.positive("max_distance");
    }
    breakForce_ = s.positive("break_force", PX_MAX_F32);
    breakTorque_ = s.positive("break_torque", PX_MAX_F32);
}

PxRigidActor* Joint::actorOf(World& w, const std::string& id) {
    return id == "world" ? nullptr : &w.entity(id, this->id()).actor();
}

void Joint::link(World& w) {
    PxRigidActor* a = actorOf(w, a_);
    PxRigidActor* b = actorOf(w, b_);
    if (!b) throw SpecError(id() + ": b must be a body, not the world");
    // Joint frame: origin at the anchor, local +X along the axis (PhysX's hinge and slide axis).
    const Pose frame{anchor_, Quat::between({1, 0, 0}, axis_)};
    const auto localFrame = [&](PxRigidActor* act) {
        return px(act ? fromPx(act->getGlobalPose()).inverse() * frame : frame);
    };
    PxPhysics& ph = w.physics();
    const PxTransform fa = localFrame(a), fb = localFrame(b);
    if (type_ == "fixed") {
        joint_.reset(PxFixedJointCreate(ph, a, fa, b, fb));
    } else if (type_ == "spherical") {
        joint_.reset(PxSphericalJointCreate(ph, a, fa, b, fb));
    } else if (type_ == "revolute") {
        auto* j = PxRevoluteJointCreate(ph, a, fa, b, fb);
        joint_.reset(j);
        if (limits_) {
            j->setLimit(PxJointAngularLimitPair(float(deg2rad(limits_->lower)), float(deg2rad(limits_->upper))));
            j->setRevoluteJointFlag(PxRevoluteJointFlag::eLIMIT_ENABLED, true);
        }
    } else if (type_ == "prismatic") {
        auto* j = PxPrismaticJointCreate(ph, a, fa, b, fb);
        joint_.reset(j);
        if (limits_) {
            j->setLimit(PxJointLinearLimitPair(ph.getTolerancesScale(), float(limits_->lower), float(limits_->upper)));
            j->setPrismaticJointFlag(PxPrismaticJointFlag::eLIMIT_ENABLED, true);
        }
    } else {
        auto* j = PxDistanceJointCreate(ph, a, fa, b, fb);
        joint_.reset(j);
        const double rest = (fromPx(b->getGlobalPose().transform(fb.p)) -
                             (a ? fromPx(a->getGlobalPose().transform(fa.p)) : anchor_)).norm();
        j->setMinDistance(float(minDistance_.value_or(0)));
        j->setMaxDistance(float(maxDistance_.value_or(std::max(rest, 1e-3))));
        j->setDistanceJointFlag(PxDistanceJointFlag::eMIN_DISTANCE_ENABLED, minDistance_.has_value());
        j->setDistanceJointFlag(PxDistanceJointFlag::eMAX_DISTANCE_ENABLED, true);
        if (spring_) {
            j->setStiffness(float(spring_->stiffness));
            j->setDamping(float(spring_->damping));
            j->setDistanceJointFlag(PxDistanceJointFlag::eSPRING_ENABLED, true);
        }
    }
    if (!joint_) throw SpecError(id() + ": PhysX could not create the joint");
    joint_->setBreakForce(float(breakForce_), float(breakTorque_));
    joint_->setConstraintFlag(PxConstraintFlag::eCOLLISION_ENABLED, false);
    if (motor_) startMotor(w, *motor_);
}

void Joint::startMotor(World& w, const Motor& m) {
    auto& j = static_cast<PxRevoluteJoint&>(*joint_);
    j.setDriveVelocity(float(m.velocity));
    j.setDriveForceLimit(float(m.maxForce));
    j.setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true);
    const auto* body = w.entity(b_, id()).actor().is<PxRigidDynamic>();
    motorId_ = id() + "/motor";
    w.actuators().declare(motorId_, "joint_motor", body ? body->getMass() : 0.0, true);
    motor_ = m;
}

void Joint::afterStep(World& w, double dt) {
    if (broken_) return;
    if (joint_->getConstraintFlags() & PxConstraintFlag::eBROKEN) {
        broken_ = true;
        joint_.reset();
        w.events().emit(id() + ":broken", w.time());
        return;
    }
    if (motor_) {
        auto& j = static_cast<PxRevoluteJoint&>(*joint_);
        PxVec3 force, torque;
        j.getConstraint()->getForce(force, torque);
        // The constraint torque is in world space; the hinge axis is the joint frame's X on actor 0.
        PxRigidActor *a0 = nullptr, *a1 = nullptr;
        j.getActors(a0, a1);
        PxTransform frame = j.getLocalPose(PxJointActorIndex::eACTOR0);
        if (a0) frame = a0->getGlobalPose() * frame;
        const double tau = torque.dot(frame.q.getBasisVector0());
        w.actuators().step(motorId_, tau, tau * j.getVelocity(), dt);
    }
}

void Joint::record(World&, Recorder& rec) {
    float value = 0;
    if (!broken_ && type_ == "revolute") value = static_cast<PxRevoluteJoint&>(*joint_).getAngle();
    if (!broken_ && type_ == "prismatic") value = static_cast<PxPrismaticJoint&>(*joint_).getPosition();
    rec.put("joint/" + id(), {broken_ ? 1.0f : 0.0f, value});
}

Json Joint::describe() const {
    return {{"type", type_},
            {"a", a_},
            {"b", b_},
            {"anchor", toJson(anchor_)},
            {"axis", toJson(axis_)},
            {"broken", broken_},
            {"streams", {{"state", "joint/" + id()}}},
            {"stream_columns", {"broken", type_ == "prismatic" ? "position_m" : "angle_rad"}}};
}

void Joint::act(World& w, const Action& a) {
    const Spec p = Spec::fromJson(a.params, id() + "." + a.verb);
    if (a.verb == "motor") {
        if (type_ != "revolute") throw SpecError(id() + ": only revolute joints have a motor");
        const double velocity = p.num("velocity");
        const double maxForce = motor_ ? p.positive("max_force", motor_->maxForce) : p.positive("max_force");
        p.rejectUnknownKeys();
        if (!broken_) startMotor(w, {velocity, maxForce});
        return;
    }
    p.rejectUnknownKeys();
    if (broken_) return;
    broken_ = true;
    joint_.reset();
    w.events().emit(id() + ":released", w.time());
}

}  // namespace p2s
