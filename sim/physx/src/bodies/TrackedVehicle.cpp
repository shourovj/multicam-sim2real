#include "bodies/TrackedVehicle.h"

using namespace physx;

namespace p2s {

namespace {
constexpr double kSideResponseS = 0.05, kYawResponseS = 0.05;
constexpr double kContactBandM = 0.05;  // tracks count as on the ground when the hull origin is this close to it
}  // namespace

TrackedVehicle::TrackedVehicle(const Spec& s, World& w) : RigidBody(s, w, "tracked_vehicle") {
    const Spec t = s["traction"];
    target_ = t.num("speed_mps");
    mu_ = t.positive("mu", 0.7);
    halfLength_ = t.positive("track_half_length_m", 1.8);
    response_ = t.positive("response_s", 0.25);
}

void TrackedVehicle::link(World& w) {
    RigidBody::link(w);
    groundZ_ = pose().p.z;
    w.actuators().declare(id() + "/tracks", "traction", mass());
}

void TrackedVehicle::beforeStep(World& w, double dt) {
    RigidBody::beforeStep(w, dt);
    const Pose p = pose();
    onGround_ = p.p.z <= groundZ_ + kContactBandM;
    lastForce_ = 0;
    if (!onGround_) {
        w.actuators().step(id() + "/tracks", 0, 0, dt);
        return;
    }
    PxRigidDynamic& d = dynamic();
    const double m = mass(), g = w.gravity().norm();
    const Vec3 fwd = p.q.axisX(), side = p.q.axisY(), v = velocity();
    const double fMax = mu_ * m * g;
    const double ff = std::clamp(m * (target_ - v.dot(fwd)) / response_, -fMax, fMax);
    const double fs = std::clamp(-m * v.dot(side) / kSideResponseS, -fMax, fMax);
    d.addForce(px(fwd * ff + side * fs));
    // Skid steering holds the heading: yaw torque limited by track friction times half the track length.
    const double wz = d.getAngularVelocity().z;
    const double izz = d.getMassSpaceInertiaTensor().z;
    d.addTorque(PxVec3(0, 0, float(std::clamp(-izz * wz / kYawResponseS, -fMax * halfLength_, fMax * halfLength_))));
    lastForce_ = ff;
    w.actuators().step(id() + "/tracks", ff, ff * v.dot(fwd), dt);
}

void TrackedVehicle::record(World& w, Recorder& rec) {
    RigidBody::record(w, rec);
    rec.put("tracked/" + id() + "/drive", {float(target_), float(lastForce_), onGround_ ? 1.0f : 0.0f});
}

Json TrackedVehicle::describe() const {
    Json j = RigidBody::describe();
    j["traction"] = {{"mu", mu_}, {"track_half_length_m", halfLength_}};
    j["streams"]["drive"] = "tracked/" + id() + "/drive";
    j["stream_columns"] = {{"drive", {"target_speed_mps", "drive_force_n", "on_ground"}}};
    return j;
}

std::vector<std::string> TrackedVehicle::verbs() const {
    auto v = RigidBody::verbs();
    v.insert(v.end(), {"brake", "set_speed"});
    return v;
}

void TrackedVehicle::act(World& w, const Action& a) {
    if (a.verb == "brake" || a.verb == "set_speed") {
        const Spec p = Spec::fromJson(a.params, id() + "." + a.verb);
        target_ = a.verb == "brake" ? 0.0 : p.num("mps");
        p.rejectUnknownKeys();
        w.events().emit(id() + ":" + a.verb, w.time());
        return;
    }
    RigidBody::act(w, a);
}

}  // namespace p2s
