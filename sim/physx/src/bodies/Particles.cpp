#include "bodies/Particles.h"

#include "extensions/PxParticleExt.h"

using namespace physx;

namespace p2s {

namespace {

constexpr double kFree = -1e9;
const PxVec4 kParked(0, 0, -1e4, 0);  // free slots wait far below the ground, motionless

struct Preset {
    const char* kind;
    double density, viscosity, cohesion, adhesion, surfaceTension, friction, damping, buoyancy, lifetime, drag;
};

// Calibrated for ~1 cm particles: oil as in the tank-truck crash (viscosity 1, cohesion 0.5, tension 0.3).
const std::map<std::string, Preset>& presets() {
    static const std::map<std::string, Preset> table = {
        {"water", {"fluid", 1000, 0.05, 0.05, 0.0, 0.02, 0.1, 0.0, 0, 1e9, 0}},
        {"oil", {"fluid", 870, 1.0, 0.5, 0.0, 0.3, 0.6, 0.02, 0, 1e9, 0}},
        {"honey", {"fluid", 1400, 8.0, 1.0, 0.5, 0.3, 0.8, 0.05, 0, 1e9, 0}},
        {"sand", {"granular", 1600, 0.0, 0.0, 0.0, 0.0, 0.8, 0.05, 0, 1e9, 0}},
        {"smoke", {"smoke", 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12, 4.0, 1.0}},
    };
    return table;
}

}  // namespace

ParticleSystem::ParticleSystem(const Spec& s, World&) : Entity(s.str("id"), "particles") {
    std::vector<std::string> names;
    for (const auto& [n, _] : presets()) names.push_back(n);
    preset_ = s.oneOf("preset", names);
    const Preset& p = presets().at(preset_);
    const std::string kind = s.oneOf("kind", {"fluid", "granular", "smoke"}, p.kind);
    kind_ = kind == "fluid" ? Kind::Fluid : kind == "granular" ? Kind::Granular : Kind::Smoke;
    spacing_ = s.positive("spacing_m", kind_ == Kind::Smoke ? 0.04 : 0.012);
    max_ = s.integer("max_particles", 30000);
    if (max_ < 1 || max_ > 2000000) s["max_particles"].fail("must be in 1..2000000");
    density_ = s.positive("density", p.density);
    viscosity_ = s.num("viscosity", p.viscosity);
    cohesion_ = s.num("cohesion", p.cohesion);
    adhesion_ = s.num("adhesion", p.adhesion);
    surfaceTension_ = s.num("surface_tension", p.surfaceTension);
    friction_ = s.num("friction", p.friction);
    damping_ = s.num("damping", p.damping);
    buoyancy_ = s.num("buoyancy", p.buoyancy);
    lifetime_ = s.positive("lifetime_s", p.lifetime);
    drag_ = s.num("drag", p.drag);
    for (const Spec& b : s.items("blocks")) {
        Block blk;
        blk.box.add(b.vec3("min"));
        blk.box.add(b.vec3("max"));
        blk.velocity = b.vec3("velocity", {0, 0, 0});
        blocks_.push_back(blk);
    }
    for (const Spec& e : s.items("emitters")) {
        Emitter em{e.str("host", ""), e.vec3("pos"), e.vec3("direction", {0, 0, -1}).normalized(),
                   e.positive("radius_m"), e.positive("speed_mps"), e.flag("active", true), {}};
        for (const Spec& ph : e.items("phases"))
            em.phases.push_back({ph.num("after_s"), ph.positive("radius_m", em.radius), ph.positive("speed_mps", em.speed)});
        emitters_.push_back(em);
    }
    if (blocks_.empty() && emitters_.empty()) s.fail("needs blocks or emitters");
}

void ParticleSystem::link(World& w) {
    PxPhysics& ph = w.physics();
    PxCudaContextManager& cuda = w.runtime().cuda();
    for (Emitter& e : emitters_)
        if (!e.host.empty()) e.hostRef = w.entity(e.host, id());
    const double gravityScale = kind_ == Kind::Smoke ? -buoyancy_ : 1.0;
    // Drag on smoke is applied here (beforeStep): PhysX's own particle drag/wind are deprecated and inert.
    material_.reset(ph.createPBDMaterial(float(friction_), float(damping_), float(adhesion_), float(viscosity_), 0.0f,
                                         float(surfaceTension_), float(cohesion_), 0.0f, 0.0f, 1.0f,
                                         float(gravityScale)));
    system_.reset(ph.createPBDParticleSystem(cuda, 96));
    if (!system_) throw std::runtime_error(id() + ": createPBDParticleSystem failed");
    // Offsets as in the tank-truck oil: fluid particles sit one spacing apart; against solids they keep a bit
    // more. Granular particles are spheres of one spacing diameter.
    const float half = float(0.5 * spacing_);
    const float rest = kind_ == Kind::Granular ? half : half / 0.6f;
    system_->setRestOffset(rest);
    system_->setContactOffset(rest + 0.005f);
    system_->setParticleContactOffset(kind_ == Kind::Granular ? 1.2f * half : rest);
    system_->setSolidRestOffset(rest);
    system_->setFluidRestOffset(half);
    system_->setParticleFlag(PxParticleFlag::eENABLE_SPECULATIVE_CCD, kind_ != Kind::Smoke);  // no leaks through thin walls
    system_->setMaxVelocity(rest * 400.0f);
    system_->setSolverIterationCounts(6, 1);
    system_->setSimulationFilterData(w.filterFor(id()).data());
    w.scene().addActor(*system_);

    PxParticlePhaseFlags flags = PxParticlePhaseFlag::eParticlePhaseSelfCollide;
    if (kind_ != Kind::Granular) flags |= PxParticlePhaseFlag::eParticlePhaseFluid;
    if (kind_ == Kind::Smoke) flags = PxParticlePhaseFlag::eParticlePhaseFluid;  // no self collision: a gas
    phase_ = system_->createPhase(material_.get(), flags);
    invMass_ = float(1.0 / (density_ * spacing_ * spacing_ * spacing_));
    pos_ = Pinned<PxVec4>(cuda, size_t(max_));
    vel_ = Pinned<PxVec4>(cuda, size_t(max_));
    Pinned<PxU32> phases(cuda, size_t(max_));
    born_.assign(size_t(max_), kFree);
    for (int i = max_ - 1; i >= 0; --i) free_.push_back(i);
    for (int i = 0; i < max_; ++i) {
        pos_[size_t(i)] = kParked;
        vel_[size_t(i)] = PxVec4(0.0f);
        phases[size_t(i)] = phase_;
    }
    for (const Block& b : blocks_) {  // a cubic lattice at the particle spacing, slightly jittered
        std::uniform_real_distribution<double> jit(-0.05 * spacing_, 0.05 * spacing_);
        for (double x = b.box.lo.x + spacing_ / 2; x < b.box.hi.x; x += spacing_)
            for (double y = b.box.lo.y + spacing_ / 2; y < b.box.hi.y; y += spacing_)
                for (double z = b.box.lo.z + spacing_ / 2; z < b.box.hi.z; z += spacing_) {
                    const int i = freeSlot();
                    if (i < 0) throw SpecError(id() + ": blocks need more than max_particles=" + std::to_string(max_));
                    const Vec3 p{x + jit(w.rng()), y + jit(w.rng()), z + jit(w.rng())};
                    pos_[size_t(i)] = PxVec4(px(p), invMass_);
                    vel_[size_t(i)] = PxVec4(px(b.velocity), 0);
                    born_[size_t(i)] = 0;
                    highWater_ = std::max(highWater_, i + 1);
                }
    }
    ExtGpu::PxParticleBufferDesc desc;
    desc.maxParticles = PxU32(max_);
    desc.numActiveParticles = PxU32(highWater_);
    desc.positions = pos_.data();
    desc.velocities = vel_.data();
    desc.phases = phases.data();
    buffer_.reset(ExtGpu::PxCreateAndPopulateParticleBuffer(desc, &cuda));
    if (!buffer_) throw std::runtime_error(id() + ": particle buffer allocation failed");
    system_->addParticleBuffer(buffer_.get());
    readBack(w);
}

int ParticleSystem::freeSlot() {
    if (free_.empty()) return -1;
    const int i = free_.back();
    free_.pop_back();
    return i;
}

bool ParticleSystem::emit(World& w, Emitter& e) {
    double radius = e.radius, speed = e.speed;
    const double age = w.time() - e.startedAt;
    for (const Phase& p : e.phases)
        if (age >= p.afterS) radius = p.radius, speed = p.speed;
    if (w.time() - e.lastLayer < spacing_ * 1.05 / speed) return false;  // one disc layer per spacing travelled
    e.lastLayer = w.time();
    Pose frame;
    Vec3 baseVel;
    if (e.hostRef) {
        const Entity& h = e.hostRef->get();
        frame = h.pose();
        baseVel = h.velocity();
    }
    const Vec3 origin = frame.transform(e.pos), dir = frame.q.rotate(e.dir);
    const Vec3 u = (std::fabs(dir.z) < 0.9 ? dir.cross({0, 0, 1}) : dir.cross({1, 0, 0})).normalized(), v = dir.cross(u);
    const int n = int(radius / spacing_);
    std::uniform_real_distribution<double> jit(-0.12 * spacing_, 0.12 * spacing_);
    bool any = false;
    for (int i = -n; i <= n; ++i)
        for (int k = -n; k <= n; ++k) {
            const double a = (i + 0.5 * (k & 1)) * spacing_, b = k * spacing_ * 0.866;  // hexagonal packing
            if (a * a + b * b > radius * radius) continue;
            const int slot = freeSlot();
            if (slot < 0) return any;
            const Vec3 p = origin + u * (a + jit(w.rng())) + v * (b + jit(w.rng())) + dir * jit(w.rng());
            pos_[size_t(slot)] = PxVec4(px(p), invMass_);
            vel_[size_t(slot)] = PxVec4(px(baseVel + dir * speed), 0);
            born_[size_t(slot)] = w.time();
            highWater_ = std::max(highWater_, slot + 1);
            emitted_ += spacing_ * spacing_ * spacing_;
            any = true;
        }
    return any;
}

void ParticleSystem::readBack(World& w) {
    PxCudaContextManager& cuda = w.runtime().cuda();
    {
        PxScopedCudaLock lock(cuda);
        PxCudaContext* cc = cuda.getCudaContext();
        cc->memcpyDtoH(pos_.data(), CUdeviceptr(buffer_->getPositionInvMasses()), size_t(highWater_) * sizeof(PxVec4));
        cc->memcpyDtoH(vel_.data(), CUdeviceptr(buffer_->getVelocities()), size_t(highWater_) * sizeof(PxVec4));
    }
    alivePoints_.clear();
    kinetic_ = potential_ = 0;
    const double m = 1.0 / invMass_;
    for (int i = 0; i < highWater_; ++i) {
        if (born_[size_t(i)] <= kFree) continue;
        const Vec3 p = fromPx(pos_[size_t(i)].getXYZ());
        alivePoints_.push_back(p);
        if (kind_ == Kind::Smoke) continue;  // a buoyant gas's energy is the air's, not the scene's
        kinetic_ += 0.5 * m * vel_[size_t(i)].getXYZ().magnitudeSquared();
        potential_ -= m * w.gravity().dot(p);
    }
    alive_ = int(alivePoints_.size());
}

void ParticleSystem::upload(World& w) {
    PxCudaContextManager& cuda = w.runtime().cuda();
    {
        PxScopedCudaLock lock(cuda);
        PxCudaContext* cc = cuda.getCudaContext();
        cc->memcpyHtoD(CUdeviceptr(buffer_->getPositionInvMasses()), pos_.data(), size_t(highWater_) * sizeof(PxVec4));
        cc->memcpyHtoD(CUdeviceptr(buffer_->getVelocities()), vel_.data(), size_t(highWater_) * sizeof(PxVec4));
    }
    buffer_->setNbActiveParticles(PxU32(highWater_));
    buffer_->raiseFlags(PxParticleBufferFlag::eUPDATE_POSITION);
    buffer_->raiseFlags(PxParticleBufferFlag::eUPDATE_VELOCITY);
}

void ParticleSystem::beforeStep(World& w, double) {
    for (Emitter& e : emitters_)
        if (e.active) {
            if (e.startedAt < 0) e.startedAt = w.time();
            dirty_ |= emit(w, e);
        }
    if (kind_ == Kind::Smoke) {  // retire old smoke; the rest relaxes toward the local (turbulent) wind
        const double relax = 1.0 - std::exp(-drag_ * w.clock().dt());
        for (int i = 0; i < highWater_; ++i) {
            if (born_[size_t(i)] <= kFree) continue;
            if (w.time() - born_[size_t(i)] > lifetime_) {
                born_[size_t(i)] = kFree;
                free_.push_back(i);
                pos_[size_t(i)] = kParked;
                vel_[size_t(i)] = PxVec4(0.0f);
            } else {
                const Vec3 rel = w.wind().at(w.time(), fromPx(pos_[size_t(i)].getXYZ())) - fromPx(vel_[size_t(i)].getXYZ());
                vel_[size_t(i)] += PxVec4(px(rel * relax), 0);
            }
        }
        dirty_ = true;
    }
    if (dirty_) upload(w);
    dirty_ = false;
}

void ParticleSystem::afterStep(World& w, double) { readBack(w); }

void ParticleSystem::record(World& w, Recorder& rec) {
    readBack(w);
    std::vector<float> out;
    out.reserve(alivePoints_.size() * 3);
    for (const Vec3& p : alivePoints_) out.insert(out.end(), {float(p.x), float(p.y), float(p.z)});
    rec.put("particles/" + id() + "/pos", out, 3, true);
}

Json ParticleSystem::describe() const {
    static const char* kinds[] = {"fluid", "granular", "smoke"};
    return {{"particle_kind", kinds[int(kind_)]},
            {"preset", preset_},
            {"spacing_m", spacing_},
            {"particle_radius_m", 0.5 * spacing_},
            {"density", density_},
            {"max_particles", max_},
            {"emitted_volume_m3", emitted_},
            {"alive_at_end", alive_},
            {"streams", {{"pos", "particles/" + id() + "/pos"}}}};
}

void ParticleSystem::act(World& w, const Action& a) {
    for (Emitter& e : emitters_) {
        e.active = a.verb == "start";
        if (e.active) e.startedAt = w.time();
    }
    w.events().emit(id() + ":" + a.verb, w.time());
}

Energy ParticleSystem::energy(const World&) const { return {kinetic_, potential_}; }

}  // namespace p2s
