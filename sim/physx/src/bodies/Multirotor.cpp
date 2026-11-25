#include "bodies/Multirotor.h"

using namespace physx;

namespace p2s {

namespace {

// Rotor positions and spins of the standard layouts; adjacent rotors spin opposite ways.
std::vector<std::pair<Vec3, double>> layout(const std::string& name, double arm, double height) {
    int n = 4;
    double offset = kPi / 4;  // quad_x: rotors at 45 degrees off the heading
    if (name == "quad_plus") offset = 0;
    if (name == "hex") n = 6, offset = 0;
    if (name == "octo") n = 8, offset = kPi / 8;
    std::vector<std::pair<Vec3, double>> out;
    for (int i = 0; i < n; ++i) {
        const double a = offset + 2 * kPi * i / n;
        out.push_back({{arm * std::cos(a), arm * std::sin(a), height}, i % 2 == 0 ? 1.0 : -1.0});
    }
    return out;
}

Vec3 vee(const Quat& rd, const Quat& r) {
    // Attitude error of geometric control (Lee, Leok, McClamroch 2010): e_R = 1/2 vee(Rd^T R - R^T Rd),
    // computed from the relative rotation q = rd^-1 r, whose vector part is sin(angle/2) * axis.
    Quat q = rd.conj() * r;
    if (q.w < 0) q = {-q.x, -q.y, -q.z, -q.w};
    return Vec3{q.x, q.y, q.z} * (2.0 * q.w);
}

}  // namespace

Multirotor::Multirotor(const Spec& s, World& w) : RigidBody(s, w, "multirotor") {
    const Spec rs = s["rotors"];
    if (rs.isArray()) {
        for (size_t i = 0; i < rs.size(); ++i) {
            const Spec r = rs.at(i);
            const double spin = r.num("spin");
            if (spin != 1 && spin != -1) r["spin"].fail("must be 1 or -1");
            rotors_.push_back({r.vec3("pos"), spin});
        }
    } else {
        for (const auto& [p, spin] : layout(rs.oneOf("layout", {"quad_x", "quad_plus", "hex", "octo"}),
                                            rs.positive("arm_m"), rs.num("height_m", 0.0)))
            rotors_.push_back({p, spin});
    }
    if (rotors_.size() < 4) rs.fail("needs at least 4 rotors");
    const Spec rp = s.has("rotor") ? s["rotor"] : Spec::fromJson(Json::object(), "rotor");
    thrustToWeight_ = rp.positive("thrust_to_weight", 2.0);
    omegaMax_ = rp.positive("max_rpm", 9000) * 2 * kPi / 60;
    km_ = rp.positive("km_over_kf_m", 0.016);  // scaled by kf in link()
    tau_ = rp.positive("tau_s", 0.03);
    const Spec c = s["controller"];
    goals_ = GoalPath(c);
    yaw_ = deg2rad(c.num("yaw_deg", 0));
    tiltMax_ = deg2rad(c.positive("tilt_max_deg", 35));
    wPos_ = c.positive("position_bandwidth", 2.2);
    wAtt_ = c.positive("attitude_bandwidth", 25);
    if (thrustToWeight_ < 1.05) rp["thrust_to_weight"].fail("must be > 1.05 or the craft cannot climb");
}

void Multirotor::link(World& w) {
    PxRigidDynamic& d = dynamic();
    const double m = d.getMass(), g = w.gravity().norm();
    const double n = double(rotors_.size());
    kf_ = thrustToWeight_ * m * g / (n * omegaMax_ * omegaMax_);
    km_ *= kf_;
    const Vec3 com = fromPx(d.getCMassLocalPose().p);
    // Mixer: [thrust, tau_x, tau_y, tau_z] = M [w_1^2 .. w_n^2]; its pseudo-inverse allocates a wrench to rotors.
    Matrix M(4, int(n));
    for (size_t i = 0; i < rotors_.size(); ++i) {
        const Vec3 r = rotors_[i].pos - com;
        M(0, int(i)) = kf_;
        M(1, int(i)) = kf_ * r.y;
        M(2, int(i)) = -kf_ * r.x;
        M(3, int(i)) = -rotors_[i].spin * km_;
        w.actuators().declare(rotorId(i), "rotor", m / n);
    }
    mixer_ = M.pseudoInverse();
    home_ = pose().p;
    startZ_ = home_.z;
    // Resting on something at the start: motors stay off until the first goal begins.
    onGround_ = !goals_.empty() && fromPx(d.getLinearVelocity()).norm() < 1e-3 && !goals_.started(0.0);
    if (!onGround_) {  // airborne at t = 0: start with the rotors already holding the weight
        for (Rotor& r : rotors_) r.omega = std::sqrt(m * g / (n * kf_));
        liftedOff_ = true;
    }
}

Multirotor::Wrench Multirotor::control(World& w, double dt) {
    PxRigidDynamic& d = dynamic();
    const Pose p = pose();
    const Vec3 v = velocity(), om = p.q.rotateInv(fromPx(d.getAngularVelocity()));
    const double m = d.getMass();
    const Reference ref = goals_.at(w.time(), p.p, v, home_);
    // PID with all three closed-loop poles at -wPos; the integral absorbs payloads hanging on joints.
    const Vec3 e = ref.pos - p.p;
    integral_ += e * dt;
    Vec3 acc = ref.acc + e * (3 * wPos_ * wPos_) + (ref.vel - v) * (3 * wPos_) + integral_ * (wPos_ * wPos_ * wPos_) -
               w.gravity();
    // Lean no further than the tilt limit: keep the vertical part, shorten the horizontal part.
    const Vec3 horiz{acc.x, acc.y, 0};
    const double maxH = std::max(acc.z, 0.0) * std::tan(tiltMax_);
    if (horiz.norm() > maxH) acc = Vec3{0, 0, acc.z} + horiz.normalized() * maxH;
    const Vec3 zd = acc.norm() > 1e-6 ? acc.normalized() : Vec3{0, 0, 1};
    const Vec3 xc{std::cos(yaw_), std::sin(yaw_), 0};
    const Vec3 yd = zd.cross(xc).normalized(), xd = yd.cross(zd);
    const Quat qd = Quat::fromAxes(xd, yd, zd);
    const Vec3 eR = vee(qd, p.q);
    const PxVec3 I = d.getMassSpaceInertiaTensor();
    const Vec3 J{I.x, I.y, I.z};
    const Vec3 torque = -(eR.cwiseMul(J) * (wAtt_ * wAtt_)) - om.cwiseMul(J) * (2 * wAtt_) + om.cross(om.cwiseMul(J));
    lastTilt_ = std::acos(std::clamp(p.q.axisZ().z, -1.0, 1.0));
    return {m * acc.dot(p.q.axisZ()), torque};
}

void Multirotor::mix(const Wrench& want, double dt) {
    const std::vector<double> sq = mixer_ * std::vector<double>{want.thrust, want.torque.x, want.torque.y, want.torque.z};
    for (size_t i = 0; i < rotors_.size(); ++i) {
        const double cmd = std::sqrt(std::clamp(sq[i], 0.0, omegaMax_ * omegaMax_));
        rotors_[i].omega += (cmd - rotors_[i].omega) * std::min(1.0, dt / tau_);
    }
}

void Multirotor::beforeStep(World& w, double dt) {
    RigidBody::beforeStep(w, dt);
    if (killed_ || (onGround_ && !goals_.started(w.time()))) {
        mix({0, {}}, dt);
    } else {
        mix(control(w, dt), dt);
    }
    PxRigidDynamic& d = dynamic();
    const Pose p = pose();
    lastThrust_ = 0;
    for (size_t i = 0; i < rotors_.size(); ++i) {
        const Rotor& r = rotors_[i];
        const double thrust = kf_ * r.omega * r.omega;
        const Vec3 z = p.q.axisZ();
        PxRigidBodyExt::addForceAtPos(d, px(z * thrust), px(p.transform(r.pos)));
        d.addTorque(px(z * (-r.spin * km_ * r.omega * r.omega)));
        w.actuators().step(rotorId(i), thrust, km_ * r.omega * r.omega * r.omega, dt);
        lastThrust_ += thrust;
    }
}

void Multirotor::afterStep(World& w, double dt) {
    RigidBody::afterStep(w, dt);
    if (!liftedOff_ && pose().p.z > startZ_ + 0.05) {
        liftedOff_ = true;
        onGround_ = false;
        w.events().emit(id() + ":lift_off", w.time());
    }
}

void Multirotor::record(World& w, Recorder& rec) {
    RigidBody::record(w, rec);
    std::vector<float> omega;
    for (const Rotor& r : rotors_) omega.push_back(float(r.omega));
    rec.put("multirotor/" + id() + "/rotor_omega", omega, int(omega.size()));
    rec.put("multirotor/" + id() + "/thrust_tilt", {float(lastThrust_), float(lastTilt_)});
}

Json Multirotor::describe() const {
    Json j = RigidBody::describe();
    Json rotors = Json::array();
    for (const Rotor& r : rotors_) rotors.push_back({{"pos", toJson(r.pos)}, {"spin", r.spin}});
    j["rotors"] = rotors;
    j["kf"] = kf_;
    j["km"] = km_;
    j["max_rpm"] = omegaMax_ * 60 / (2 * kPi);
    j["thrust_to_weight"] = thrustToWeight_;
    j["goals"] = goals_.toJson();
    j["streams"]["rotor_omega"] = "multirotor/" + id() + "/rotor_omega";
    j["streams"]["thrust_tilt"] = "multirotor/" + id() + "/thrust_tilt";
    return j;
}

std::vector<std::string> Multirotor::verbs() const {
    auto v = RigidBody::verbs();
    v.insert(v.end(), {"kill", "goto"});
    return v;
}

void Multirotor::act(World& w, const Action& a) {
    if (a.verb == "kill") {
        killed_ = true;
        w.events().emit(id() + ":kill", w.time());
        return;
    }
    if (a.verb == "goto") {
        const Spec p = Spec::fromJson(a.params, id() + ".goto");
        const Vec3 target = p.vec3("pos");
        const Json goals = {{"goals", {{{"from_t", w.time()}, {"t", w.time() + p.positive("t_s")}, {"pos", toJson(target)}}}}};
        goals_ = GoalPath(Spec::fromJson(goals, id() + ".goto"));
        onGround_ = false;
        return;
    }
    RigidBody::act(w, a);
}

}  // namespace p2s
