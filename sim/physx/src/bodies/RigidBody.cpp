#include "bodies/RigidBody.h"

using namespace physx;

namespace p2s {

namespace {

double smoothstep(double u) {
    u = std::clamp(u, 0.0, 1.0);
    return u * u * (3 - 2 * u);
}

}  // namespace

Pose interpolateKeys(const std::vector<PoseKey>& keys, double t) {
    if (t <= keys.front().t) return keys.front().pose;
    if (t >= keys.back().t) return keys.back().pose;
    size_t i = 0;
    while (keys[i + 1].t < t) ++i;
    const PoseKey &a = keys[i], &b = keys[i + 1];
    const double u = smoothstep((t - a.t) / (b.t - a.t));
    return {a.pose.p + (b.pose.p - a.pose.p) * u, slerp(a.pose.q, b.pose.q, u)};
}

std::vector<PoseKey> parseKeys(const Spec& s, const std::string& key) {
    std::vector<PoseKey> keys;
    for (const Spec& k : s.items(key)) {
        keys.push_back({k.num("t"), k.pose()});
        if (keys.size() > 1 && keys.back().t <= keys[keys.size() - 2].t) k["t"].fail("key times must increase");
    }
    if (!keys.empty() && keys.size() < 2) s[key].fail("needs at least two keys");
    return keys;
}

RigidBody::RigidBody(const Spec& s, World& w, std::string kind) : Entity(s.str("id"), std::move(kind)) {
    type_ = s.oneOf("type", {"dynamic", "kinematic", "static"}, "dynamic");
    for (const Spec& sh : s.items("shapes"))
        for (Shape& x : Shape::parse(sh)) shapes_.push_back(std::move(x));
    if (shapes_.empty()) s.fail("needs at least one entry in shapes");
    keys_ = parseKeys(s);
    if (!keys_.empty() && type_ != "kinematic") s["keys"].fail("keys need type kinematic");
    const Pose start = keys_.empty() ? s.pose() : keys_.front().pose;
    const CollisionFilter filter = w.filterFor(id(), s.str("assembly", ""));

    if (type_ == "static") {
        actor_.reset(w.physics().createRigidStatic(px(start)));
        if (!actor_) s.fail("PhysX could not create the body (invalid pose?)");
        w.shapeFactory().attach(*actor_, shapes_, filter, id());
    } else {
        PxRigidDynamic* d = w.physics().createRigidDynamic(px(start));
        if (!d) s.fail("PhysX could not create the body (invalid pose?)");
        actor_.reset(d);
        w.shapeFactory().attach(*actor_, shapes_, filter, id());
        if (s.has("mass_kg")) {
            PxRigidBodyExt::setMassAndUpdateInertia(*d, float(s.positive("mass_kg")));
        } else if (s.has("density")) {
            PxRigidBodyExt::updateMassAndInertia(*d, float(s.positive("density")));
        } else {  // each shape weighs what its material's density says
            std::vector<PxReal> densities;
            for (const Shape& sh : shapes_) densities.push_back(float(w.materials().props(sh.material).density));
            PxRigidBodyExt::updateMassAndInertia(*d, densities.data(), PxU32(densities.size()));
        }
        if (type_ == "kinematic") d->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
        d->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_GYROSCOPIC_FORCES, true);
        d->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_CCD, s.flag("ccd", false));
        if (type_ == "dynamic") {
            d->setLinearVelocity(px(s.vec3("velocity", {0, 0, 0})));
            d->setAngularVelocity(px(s.vec3("angular_velocity", {0, 0, 0})));
        }
        d->setMaxAngularVelocity(1000.0f);
        if (s.has("damping")) {
            d->setLinearDamping(float(s["damping"].num("linear", 0)));
            d->setAngularDamping(float(s["damping"].num("angular", 0.05)));
        }
        d->setMaxDepenetrationVelocity(float(s.positive("max_depenetration_mps", 2.0)));
        const int posIters = s.integer("solver_iterations", 8);
        d->setSolverIterationCounts(PxU32(std::clamp(posIters, 1, 255)), 2);
        d->setSleepThreshold(0.0f);
        lastVel_ = velocity();
    }
    if (s.has("drag")) {
        const Spec dr = s["drag"];
        linearDrag_ = dr.num("linear_per_s", 0);
        cd_ = dr.num("cd", 0);
        if (cd_ > 0) {
            Aabb b;
            for (const Shape& sh : shapes_) {
                const Aabb sb = sh.bounds().transformed(sh.local);
                b.add(sb.lo);
                b.add(sb.hi);
            }
            const Vec3 e = b.extent();  // default frontal area: the largest face of the bounding box
            area_ = dr.positive("area_m2", std::max({e.x * e.y, e.y * e.z, e.x * e.z}));
        }
    }
    w.scene().addActor(*actor_);
}

RigidBody::RigidBody(std::string id, std::vector<Shape> shapes, const Pose& pose, World& w, uint32_t category)
    : Entity(std::move(id), "rigid_body"), type_("static"), shapes_(std::move(shapes)) {
    actor_.reset(w.physics().createRigidStatic(px(pose)));
    w.shapeFactory().attach(*actor_, shapes_, w.filterFor(this->id(), "", category), this->id());
    w.scene().addActor(*actor_);
}

PxRigidDynamic& RigidBody::dynamic() {
    auto* d = actor_->is<PxRigidDynamic>();
    if (!d) throw SpecError(id() + " is static");
    return *d;
}

double RigidBody::mass() const {
    const auto* d = actor_->is<PxRigidDynamic>();
    return d ? d->getMass() : 0.0;
}

Vec3 RigidBody::velocity() const {
    const auto* d = actor_->is<PxRigidDynamic>();
    return d ? fromPx(d->getLinearVelocity()) : Vec3{};
}

void RigidBody::applyDrag(World& w) {
    PxRigidDynamic& d = dynamic();
    const Vec3 v = fromPx(d.getLinearVelocity());
    Vec3 f = v * (-linearDrag_ * d.getMass());
    if (cd_ > 0) {
        const Vec3 rel = w.wind().at(w.time(), pose().p) - v;
        f += rel * (0.5 * w.wind().airDensity() * cd_ * area_ * rel.norm());
    }
    d.addForce(px(f));
}

void RigidBody::beforeStep(World& w, double dt) {
    if (type_ == "static") return;
    PxRigidDynamic& d = dynamic();
    if (type_ == "kinematic") {
        if (!keys_.empty()) d.setKinematicTarget(px(interpolateKeys(keys_, w.time() + dt)));
        return;
    }
    if (linearDrag_ > 0 || cd_ > 0) applyDrag(w);
    if (push_ && w.time() < push_->until) d.addForce(px(push_->force));
}

void RigidBody::afterStep(World& w, double dt) {
    if (type_ != "dynamic") return;
    const Vec3 v = velocity();
    // Acceleration beyond free fall: what the body's contents would feel.
    const double a = ((v - lastVel_) / dt - w.gravity()).norm();
    peakAccel_ = std::max(peakAccel_, a);
    peakSpeed_ = std::max(peakSpeed_, v.norm());
    lastVel_ = v;
}

void RigidBody::record(World&, Recorder& rec) {
    const Pose p = pose();
    rec.put("body/" + id() + "/pose",
            {float(p.p.x), float(p.p.y), float(p.p.z), float(p.q.x), float(p.q.y), float(p.q.z), float(p.q.w)});
    if (type_ != "static") {
        const auto& d = *actor_->is<PxRigidDynamic>();
        const PxVec3 v = d.getLinearVelocity(), om = d.getAngularVelocity();
        rec.put("body/" + id() + "/vel", {v.x, v.y, v.z, om.x, om.y, om.z});
    }
}

Json RigidBody::describe() const {
    Json shapes = Json::array();
    for (const Shape& s : shapes_) shapes.push_back(s.toJson());
    Json j = {{"type", type_},
              {"parts", {{{"name", "body"}, {"shapes", shapes}, {"stream", "body/" + id() + "/pose"}, {"row", 0}}}},
              {"streams", {{"pose", "body/" + id() + "/pose"}}}};
    if (type_ != "static") {
        j["mass_kg"] = mass();
        j["streams"]["vel"] = "body/" + id() + "/vel";
        j["peak_accel_mps2"] = peakAccel_;
        j["peak_speed_mps"] = peakSpeed_;
    }
    return j;
}

std::vector<std::string> RigidBody::verbs() const {
    if (type_ == "static") return {};
    return {"impulse", "push", "set_velocity", "release", "freeze"};
}

void RigidBody::act(World& w, const Action& a) {
    PxRigidDynamic& d = dynamic();
    const Spec p = Spec::fromJson(a.params, id() + "." + a.verb);
    const bool forceVerb = a.verb == "impulse" || a.verb == "push" || a.verb == "set_velocity";
    if (forceVerb && type_ != "dynamic")
        throw SpecError(id() + ": '" + a.verb + "' at t=" + std::to_string(w.time()) +
                        " needs a dynamic body; it is kinematic (trigger 'release' first)");
    if (a.verb == "impulse") {  // N s, at the centre of mass or at a world point
        const Vec3 j = p.vec3("vector");
        if (p.has("point"))
            PxRigidBodyExt::addForceAtPos(d, px(j), px(p.vec3("point")), PxForceMode::eIMPULSE);
        else
            d.addForce(px(j), PxForceMode::eIMPULSE);
    } else if (a.verb == "push") {  // constant force for a while
        push_ = Push{p.vec3("force"), w.time() + p.positive("duration_s")};
    } else if (a.verb == "set_velocity") {
        d.setLinearVelocity(px(p.vec3("velocity")));
    } else if (a.verb == "release") {  // kinematic -> dynamic: falls, gets knocked about
        d.setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, false);
        type_ = "dynamic";
        lastVel_ = velocity();
    } else if (a.verb == "freeze") {
        d.setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
        type_ = "kinematic";
        keys_.clear();
    }
    p.rejectUnknownKeys();
    w.events().emit(id() + ":" + a.verb, w.time());
}

Energy RigidBody::energy(const World& w) const {
    const auto* d = actor_->is<PxRigidDynamic>();
    if (!d || type_ != "dynamic") return {};
    const double m = d->getMass();
    const PxTransform com = d->getGlobalPose() * d->getCMassLocalPose();
    const PxVec3 om = com.q.rotateInv(d->getAngularVelocity());
    const PxVec3 I = d->getMassSpaceInertiaTensor();
    const double rot = 0.5 * (I.x * om.x * om.x + I.y * om.y * om.y + I.z * om.z * om.z);
    return {0.5 * m * fromPx(d->getLinearVelocity()).norm2() + rot, -m * w.gravity().dot(fromPx(com.p))};
}

std::unique_ptr<Entity> makeGround(const Spec& s, World& w) {
    // A large thin box rather than an infinite plane, so the checker and renderer share its geometry.
    const double h = s.num("height", 0.0), half = s.positive("half_size_m", 2000);
    Shape slab = Shape::box({half, half, 1.0}, Pose{{0, 0, -1.0}, {}});
    slab.material = s.str("material", "concrete");
    return std::make_unique<RigidBody>("ground", std::vector<Shape>{slab}, Pose{{0, 0, h}, {}}, w, 2);
}

}  // namespace p2s
