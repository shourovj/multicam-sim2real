#include "bodies/Explosion.h"

#include "bodies/Particles.h"
#include "bodies/RigidBody.h"

using namespace physx;

namespace p2s {

namespace {

// Area of a world box seen from direction n.
double facingArea(const Aabb& b, const Vec3& n) {
    const Vec3 e = b.extent();
    return e.y * e.z * std::fabs(n.x) + e.x * e.z * std::fabs(n.y) + e.x * e.y * std::fabs(n.z);
}

}  // namespace

Explosion::Explosion(const Spec& s, World&) : Entity(s.str("id"), "explosion") {
    pos_ = s.vec3("pos");
    if (s.has("at_s")) at_ = s.num("at_s");
    radius_ = s.positive("radius_m", 8);
    impulse_ = s.num("impulse_ns", 4000);
    if (impulse_ < 0) s["impulse_ns"].fail("must be >= 0");
    particleSpeed_ = s.num("particle_speed_mps", 12);
}

void Explosion::link(World& w) {
    w.actuators().declare(id(), "explosion", 0);
}

void Explosion::beforeStep(World& w, double dt) {
    if (detonatedAt_) {
        w.actuators().step(id(), 0, 0, dt);
        return;
    }
    if (at_ && w.time() + 1e-9 >= *at_) detonate(w, dt);
}

void Explosion::act(World& w, const Action& a) {
    Spec::fromJson(a.params, id() + "." + a.verb).rejectUnknownKeys();
    if (!detonatedAt_) at_ = w.time();  // at the next step, so every entity sees the same moment
}

void Explosion::detonate(World& w, double dt) {
    detonatedAt_ = w.time();
    double work = 0;
    for (const auto& e : w.entities()) {
        if (auto* ps = dynamic_cast<ParticleSystem*>(e.get())) {
            work += ps->kick(w, pos_, radius_, particleSpeed_);
            continue;
        }
        if (!e->isRigid() || impulse_ <= 0) continue;
        auto* d = e->actor().is<PxRigidDynamic>();
        if (!d || (d->getRigidBodyFlags() & PxRigidBodyFlag::eKINEMATIC)) continue;
        const Aabb box = boundsOf(e->shapes(), e->pose());
        if (box.empty()) continue;
        const Vec3 nearest = box.clamp(pos_);
        Vec3 away = nearest - pos_;
        const double dist = away.norm();
        if (dist > radius_) continue;
        away = dist > 1e-6 ? away / dist : (box.center() - pos_).normalized();
        if (away.norm2() < 0.5) away = {0, 0, 1};
        const Vec3 J = away * (impulse_ * facingArea(box, away) / std::pow(std::max(dist, 1.0), 2));
        work += impulseWork(*d, J, nearest);
        PxRigidBodyExt::addForceAtPos(*d, px(J), px(nearest), PxForceMode::eIMPULSE);
        peakImpulse_ = std::max(peakImpulse_, J.norm());
    }
    w.actuators().step(id(), peakImpulse_, work / dt, dt);
    w.events().emit(id() + ":detonated", w.time());
}

Json Explosion::describe() const {
    return {{"pos", toJson(pos_)},
            {"radius_m", radius_},
            {"impulse_ns", impulse_},
            {"detonated_at_s", detonatedAt_ ? Json(*detonatedAt_) : Json(nullptr)},
            {"peak_body_impulse_ns", peakImpulse_}};
}

}  // namespace p2s
