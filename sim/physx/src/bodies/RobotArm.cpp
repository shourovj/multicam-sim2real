#include "bodies/RobotArm.h"

#include <cstdlib>

#include "bodies/RigidBody.h"

using namespace physx;

namespace p2s {

namespace {

double smoothstep(double u) {
    u = std::clamp(u, 0.0, 1.0);
    return u * u * (3 - 2 * u);
}

std::string robotPath(const std::string& name) {
    const char* dir = std::getenv("P2S_ROBOTS");
    return std::string(dir ? dir : P2S_ROBOT_DIR) + "/" + name + ".json";
}

// PhysX articulation joints turn about (or slide along) their frame's X axis.
Quat xTo(const Vec3& axis) { return Quat::between({1, 0, 0}, axis); }

}  // namespace

RobotArm::RobotArm(const Spec& s, World&) : Entity(s.str("id"), "robot_arm") {
    robotName_ = s.str("robot");
    for (char c : robotName_)
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') s["robot"].fail("robot names are letters, digits and _");
    chain_ = Chain::load(robotPath(robotName_));
    base_ = s["base"].pose();
    pointLocal_ = s.vec3("point_local", {0, 0, 0});
    if (s.has("hold")) {
        const Spec h = s["hold"];
        holdId_ = h.str("object");
        holdOffset_ = h.has("offset") ? h["offset"].pose() : Pose{};
        gripper_ = h.positive("gripper_m", 0.04);
    }
    for (const Spec& k : s.items("keys")) {
        keys_.push_back({k.num("t"), k.vec3("point"), deg2rad(k.num("yaw_deg", 0)), deg2rad(k.num("tilt_deg", 0))});
        if (keys_.size() > 1 && keys_.back().t <= keys_[keys_.size() - 2].t) k["t"].fail("key times must increase");
    }
    if (keys_.empty()) s.fail("needs keys");
    if (s.has("drive")) {
        stiffness_ = s["drive"].positive("stiffness", stiffness_);
        damping_ = s["drive"].positive("damping", damping_);
        maxTorque_ = s["drive"].positive("max_torque_nm", maxTorque_);
    }
}

RobotArm::~RobotArm() {
    grip_.reset();
    cache_.reset();
}

PxRigidActor& RobotArm::actor() { return links_[size_t(chain_.tcpLink())].get(); }

Pose RobotArm::targetTcp(double t) const {
    size_t i = 0;
    while (i + 2 < keys_.size() && keys_[i + 1].t < t) ++i;
    const Key &a = keys_[i], &b = keys_[std::min(i + 1, keys_.size() - 1)];
    const double u = b.t > a.t ? smoothstep((t - a.t) / (b.t - a.t)) : 0.0;
    const Vec3 point = a.point + (b.point - a.point) * u;
    const Quat R = Quat::axisAngle({0, 0, 1}, a.yaw + (b.yaw - a.yaw) * u) * Quat::axisAngle({0, 1, 0}, a.tilt + (b.tilt - a.tilt) * u);
    const Pose object{point - R.rotate(pointLocal_), R};  // the keyed point sits at `point`
    return holdId_.empty() ? object : object * holdOffset_.inverse();
}

void RobotArm::link(World& w) {
    const auto& models = chain_.links();
    std::vector<double> q(models.size(), 0.0);
    for (size_t i = 0; i < models.size(); ++i)
        if (models[i].joint && !models[i].joint->revolute) q[i] = gripper_;
    const auto sol = chain_.solve(base_, targetTcp(0), chain_.home(), q, 400);
    armQ_ = sol.arm;
    q = chain_.fullQ(armQ_, q);
    const auto poses = chain_.forward(base_, q);

    articulation_.reset(w.physics().createArticulationReducedCoordinate());
    articulation_->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
    articulation_->setSolverIterationCounts(32, 4);
    const CollisionFilter filter = w.filterFor(id(), id());
    for (size_t i = 0; i < models.size(); ++i) {
        const LinkModel& m = models[i];
        PxArticulationLink* parent = m.parent < 0 ? nullptr : &links_[size_t(m.parent)].get();
        PxArticulationLink* link = articulation_->createLink(parent, px(poses[i]));
        if (!link) throw std::runtime_error(id() + ": createLink failed");
        std::vector<Shape> shapes = m.shapes;
        w.shapeFactory().attach(*link, shapes, filter, id());
        PxRigidBodyExt::setMassAndUpdateInertia(*link, float(std::max(m.massKg, 0.01)));
        link->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);  // the arm's controller compensates gravity
        if (parent) {
            auto* j = static_cast<PxArticulationJointReducedCoordinate*>(link->getInboundJoint());
            const Vec3 axis = m.joint ? m.joint->axis : Vec3{1, 0, 0};
            j->setParentPose(px(m.local * Pose{{}, xTo(axis)}));
            j->setChildPose(px(Pose{{}, xTo(axis)}));
            if (!m.joint) {
                j->setJointType(PxArticulationJointType::eFIX);
            } else {
                const bool rev = m.joint->revolute;
                const PxArticulationAxis::Enum ax = rev ? PxArticulationAxis::eTWIST : PxArticulationAxis::eX;
                j->setJointType(rev ? PxArticulationJointType::eREVOLUTE : PxArticulationJointType::ePRISMATIC);
                j->setMotion(ax, PxArticulationMotion::eLIMITED);
                j->setLimitParams(ax, PxArticulationLimit(float(m.joint->lower), float(m.joint->upper)));
                j->setDriveParams(ax, PxArticulationDrive(float(stiffness_ * (rev ? 1 : 50)), float(damping_ * (rev ? 1 : 5)),
                                                          float(rev ? maxTorque_ : 70.0), PxArticulationDriveType::eFORCE));
                j->setDriveTarget(ax, float(q[i]));
                j->setJointPosition(ax, float(q[i]));
            }
        }
        links_.push_back(*link);
    }
    w.scene().addArticulation(*articulation_);
    cache_.reset(articulation_->createCache());

    // Cache layout: links in PhysX's internal order, each with its degrees of freedom.
    std::vector<std::pair<PxU32, size_t>> order;
    for (size_t i = 0; i < links_.size(); ++i) order.push_back({links_[i].get().getLinkIndex(), i});
    std::sort(order.begin(), order.end());
    dofStart_.assign(links_.size(), -1);
    int next = 0;
    for (const auto& [_, i] : order) {
        const int dofs = int(links_[i].get().getInboundJointDof());
        if (dofs > 0) dofStart_[i] = next;
        next += std::max(dofs, 0);
    }
    q_ = q;
    qd_.assign(q.size(), 0.0);

    if (!holdId_.empty()) {  // place the held object in the gripper and fix it there
        Entity& obj = w.entity(holdId_, id());
        auto* body = obj.actor().is<PxRigidDynamic>();
        if (!body) throw SpecError(id() + ": hold.object '" + holdId_ + "' must be a dynamic body");
        const Pose tcp = chain_.tcp(base_, q);
        body->setGlobalPose(px(tcp * holdOffset_));
        setFilter(*body, w.filterFor(holdId_, id()));  // the object never collides with the hand holding it
        PxArticulationLink& hand = links_[size_t(chain_.tcpLink())].get();
        grip_.reset(PxFixedJointCreate(w.physics(), &hand, px(chain_.tcpOffset() * holdOffset_), body, PxTransform(PxIdentity)));
        if (!grip_) throw std::runtime_error(id() + ": could not create the grip joint");
    }
    w.actuators().declare(id() + "/joints", "joint_motor", 0.0, true);
}

void RobotArm::readJoints() {
    articulation_->copyInternalStateToCache(*cache_, PxArticulationCacheFlag::ePOSITION | PxArticulationCacheFlag::eVELOCITY);
    for (size_t i = 0; i < links_.size(); ++i)
        if (dofStart_[i] >= 0) {
            q_[i] = cache_->jointPosition[dofStart_[i]];
            qd_[i] = cache_->jointVelocity[dofStart_[i]];
        }
}

void RobotArm::beforeStep(World& w, double dt) {
    readJoints();
    const auto& models = chain_.links();
    const auto sol = chain_.solve(base_, targetTcp(w.time() + dt), armQ_, q_, 60);
    armQ_ = sol.arm;
    ikErrM_ = sol.errorM;
    ikErrDeg_ = sol.errorDeg;
    const auto target = chain_.fullQ(armQ_, q_);
    double power = 0, peak = 0;
    for (size_t i = 0; i < models.size(); ++i) {
        if (!models[i].joint || !models[i].joint->revolute) continue;
        auto* j = static_cast<PxArticulationJointReducedCoordinate*>(links_[i].get().getInboundJoint());
        j->setDriveTarget(PxArticulationAxis::eTWIST, float(target[i]));
        // What the drive applies: a stiff spring-damper to the target, capped at the joint's torque limit.
        const double tau = std::clamp(stiffness_ * (target[i] - q_[i]) - damping_ * qd_[i], -maxTorque_, maxTorque_);
        power += tau * qd_[i];
        peak = std::max(peak, std::fabs(tau));
    }
    w.actuators().step(id() + "/joints", peak, power, dt);
}

void RobotArm::record(World& w, Recorder& rec) {
    readJoints();
    std::vector<float> poses;
    for (const auto& l : links_) {
        const Pose p = fromPx(l.get().getGlobalPose());
        poses.insert(poses.end(), {float(p.p.x), float(p.p.y), float(p.p.z), float(p.q.x), float(p.q.y), float(p.q.z), float(p.q.w)});
    }
    rec.put("robot/" + id() + "/links", poses, 7);
    std::vector<float> joints;
    for (double v : q_) joints.push_back(float(v));
    joints.insert(joints.end(), {float(ikErrM_), float(ikErrDeg_)});
    rec.put("robot/" + id() + "/joints", joints, int(joints.size()));
    (void)w;
}

Json RobotArm::describe() const {
    Json parts = Json::array(), names = Json::array();
    const auto& models = chain_.links();
    for (size_t i = 0; i < models.size(); ++i) {
        Json shapes = Json::array();
        for (const Shape& s : models[i].shapes) shapes.push_back(s.toJson());
        parts.push_back({{"name", models[i].name}, {"shapes", shapes}, {"stream", "robot/" + id() + "/links"},
                         {"row", i}, {"kind", "robot_link"}, {"visuals", models[i].visuals}});
        names.push_back(models[i].name);
    }
    return {{"robot", robotName_},
            {"parts", parts},
            {"links", names},
            {"hold", holdId_},
            {"base", {{"pos", toJson(base_.p)}, {"quat_xyzw", toJson(base_.q)}}},
            {"streams", {{"links", "robot/" + id() + "/links"}, {"joints", "robot/" + id() + "/joints"}}},
            {"stream_columns", {{"joints", "one value per link (rad or m; 0 without a joint), then ik_error_m, ik_error_deg"}}}};
}

Energy RobotArm::energy(const World&) const {
    double ke = 0;  // links feel no gravity (compensated), so they have no potential energy to account
    for (const auto& l : links_) {
        const PxArticulationLink& link = l.get();
        const PxVec3 v = link.getLinearVelocity(), om = link.getAngularVelocity();
        const PxVec3 I = link.getMassSpaceInertiaTensor();
        const PxVec3 omL = (link.getGlobalPose() * link.getCMassLocalPose()).q.rotateInv(om);
        ke += 0.5 * link.getMass() * v.magnitudeSquared() + 0.5 * (I.x * omL.x * omL.x + I.y * omL.y * omL.y + I.z * omL.z * omL.z);
    }
    return {ke, 0};
}

}  // namespace p2s
