#include "bodies/Magnet.h"

using namespace physx;

namespace p2s {

namespace {

PxRigidDynamic* movable(Entity& e) {
    auto* d = e.actor().is<PxRigidDynamic>();
    return d && !(d->getRigidBodyFlags() & PxRigidBodyFlag::eKINEMATIC) ? d : nullptr;
}

}  // namespace

Magnet::Magnet(const Spec& s, World&) : Entity(s.str("id"), "magnet") {
    host_ = s.str("host", "world");
    pos_ = s.vec3("pos");
    targetIds_ = s.strings("targets");
    if (targetIds_.empty()) s["targets"].fail("needs at least one body");
    k_ = s.positive("force_n_at_1m");
    max_ = s.positive("max_force_n", 10 * k_);
    falloff_ = s.positive("falloff", 2);
    repel_ = s.flag("repel", false);
    active_ = s.flag("active", true);
}

void Magnet::link(World& w) {
    if (host_ != "world") {
        Entity& h = w.entity(host_, id() + ".host");
        if (!h.isRigid()) throw SpecError(id() + ".host: '" + host_ + "' is not a rigid body");
        hostRef_ = h;
    }
    for (const std::string& t : targetIds_) {
        Entity& e = w.entity(t, id() + ".targets");
        if (!e.isRigid()) throw SpecError(id() + ".targets: '" + t + "' is not a rigid body");
        if (t == host_) throw SpecError(id() + ".targets: the host cannot attract itself");
        targets_.push_back(e);
    }
    w.actuators().declare(id(), "magnet", 0);
}

Vec3 Magnet::worldPos() const { return hostRef_ ? hostRef_->get().pose().transform(pos_) : pos_; }

void Magnet::beforeStep(World& w, double dt) {
    double power = 0, strongest = 0;
    if (active_) {
        const Vec3 m = worldPos();
        PxRigidDynamic* host = hostRef_ ? movable(hostRef_->get()) : nullptr;
        for (Entity& t : targets_) {
            const Aabb box = boundsOf(t.shapes(), t.pose());
            const Vec3 near = box.clamp(m);
            Vec3 toward = box.center() - m;  // direction from magnet to target
            const double d = std::max((near - m).norm(), 0.02);
            if (toward.norm2() < 1e-12) continue;
            toward = toward.normalized();
            const double f = std::min(k_ / std::pow(d, falloff_), max_);
            const Vec3 F = toward * (repel_ ? f : -f);  // on the target
            if (PxRigidDynamic* td = movable(t)) {
                td->addForce(px(F));
                power += F.dot(t.velocity());
            }
            if (host) {
                PxRigidBodyExt::addForceAtPos(*host, px(-F), px(m));
                power -= F.dot(hostRef_->get().velocity());
            }
            strongest = std::max(strongest, f);
        }
    }
    peak_ = std::max(peak_, strongest);
    w.actuators().step(id(), strongest, power, dt);
}

void Magnet::act(World& w, const Action& a) {
    Spec::fromJson(a.params, id() + "." + a.verb).rejectUnknownKeys();
    if (a.verb == "on") active_ = true;
    if (a.verb == "off") active_ = false;
    if (a.verb == "repel") repel_ = true;
    if (a.verb == "attract") repel_ = false;
    w.events().emit(id() + ":" + a.verb, w.time());
}

void Magnet::record(World&, Recorder& rec) {
    const Vec3 p = worldPos();
    rec.put("magnet/" + id(), {float(p.x), float(p.y), float(p.z), active_ ? (repel_ ? -1.0f : 1.0f) : 0.0f});
}

Json Magnet::describe() const {
    return {{"host", host_},
            {"targets", targetIds_},
            {"force_n_at_1m", k_},
            {"max_force_n", max_},
            {"peak_force_n", peak_},
            {"streams", {{"state", "magnet/" + id()}}},
            {"stream_columns", {"x", "y", "z", "polarity"}}};
}

}  // namespace p2s
