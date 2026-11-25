#include "world/World.h"

#include <chrono>
#include <filesystem>
#include <fstream>

#include "bodies/RigidBody.h"
#include "core/Camera.h"

using namespace physx;

namespace p2s {

namespace {

using Clk = std::chrono::steady_clock;
double since(Clk::time_point a) { return std::chrono::duration<double>(Clk::now() - a).count(); }

Aabb worldBounds(const Entity& e) {
    Aabb box;
    const Pose p = e.pose();
    for (const Shape& s : e.shapes()) {
        const Aabb b = s.bounds();
        for (int i = 0; i < 8; ++i)
            box.add((p * s.local).transform({i & 1 ? b.hi.x : b.lo.x, i & 2 ? b.hi.y : b.lo.y, i & 4 ? b.hi.z : b.lo.z}));
    }
    return box;
}

double aabbGap(const Aabb& a, const Aabb& b) {
    const Vec3 gap = (a.lo - b.hi).cwiseMax(b.lo - a.hi).cwiseMax({0, 0, 0});
    return gap.norm();
}

}  // namespace

const World::FactoryList& World::factories() {
    static const FactoryList list = builtinEntities();
    return list;
}

World::World(Runtime& runtime, const Spec& scene)
    : runtime_(runtime),
      materials_(runtime.physics()),
      shapeFactory_(runtime, materials_),
      contacts_(events_, [this](uint32_t i) { return entityName(i); }) {
    build(scene);
}

World::~World() {
    cameras_.clear();
    entities_.clear();  // actors and joints go before the scene
    scene_.reset();
}

std::string World::entityName(uint32_t index) const {
    return index > 0 && index <= entities_.size() ? entities_[index - 1]->id() : "unknown";
}

Entity& World::entity(const std::string& id, const std::string& usedBy) {
    const auto it = index_.find(id);
    if (it == index_.end()) throw SpecError(usedBy + ": no entity with id '" + id + "'");
    return *entities_[it->second];
}

CollisionFilter World::filterFor(const std::string& entityId, const std::string& assembly, uint32_t category,
                                 uint32_t ignores) {
    CollisionFilter f;
    f.group = category;
    f.ignores = ignores;
    const auto it = index_.find(entityId);
    f.entity = it == index_.end() ? uint32_t(entities_.size() + 1) : uint32_t(it->second + 1);
    if (!assembly.empty()) f.assemblyId = assemblies_.try_emplace(assembly, uint32_t(assemblies_.size() + 1)).first->second;
    return f;
}

void World::build(const Spec& s) {
    if (s.integer("p2s_version", 1) != 1) s["p2s_version"].fail("only version 1 is supported");
    sceneInfo_ = {{"name", s.str("name")}};
    if (s.has("meta")) sceneInfo_["meta"] = s["meta"].raw();
    for (const char* passthrough : {"meta", "expect", "acknowledge"})  // read by the checker, not the simulator
        if (s.has(passthrough)) (void)s[passthrough].raw();

    const Spec c = s["clock"];
    clock_.fps = c.positive("fps", 24);
    const double duration = c.positive("duration_s");
    clock_.frames = int(std::lround(duration * clock_.fps)) + 1;
    clock_.substeps = c.integer("substeps", 20);
    clock_.prerollS = c.num("preroll_s", 0);
    if (clock_.substeps < 1 || clock_.substeps > 1000) c["substeps"].fail("must be in 1..1000");
    if (clock_.prerollS < 0) c["preroll_s"].fail("must be >= 0");

    const Spec w = s.has("world") ? s["world"] : Spec::fromJson(Json::object(), "world");
    gravity_ = w.vec3("gravity", {0, 0, -9.81});
    rng_.seed(unsigned(w.integer("seed", 0)));
    if (w.has("wind")) wind_ = WindField(w["wind"], unsigned(w.integer("seed", 0)) + 101);

    for (const Spec& m : s.items("materials")) materials_.add(m);

    PxSceneDesc sd(runtime_.physics().getTolerancesScale());
    sd.gravity = px(gravity_);
    sd.cpuDispatcher = &runtime_.dispatcher();
    sd.cudaContextManager = &runtime_.cuda();
    sd.filterShader = filterShader;
    sd.simulationEventCallback = &contacts_;
    sd.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS | PxSceneFlag::eENABLE_PCM | PxSceneFlag::eENABLE_CCD;
    sd.broadPhaseType = PxBroadPhaseType::eGPU;
    sd.solverType = PxSolverType::eTGS;
    sd.gpuMaxNumPartitions = 8;
    scene_.reset(runtime_.physics().createScene(sd));
    if (!scene_) throw std::runtime_error("createScene failed");

    const bool ground = !w.has("ground") || !w["ground"].raw().is_boolean() || w.flag("ground", true);
    if (ground) {
        index_["ground"] = 0;
        entities_.push_back(makeGround(w.has("ground") && w["ground"].isObject() ? w["ground"]
                                                                                 : Spec::fromJson(Json::object(), "ground"),
                                       *this));
    }
    for (const auto& [key, factory] : factories())
        for (const Spec& e : s.items(key)) {
            const std::string id = e.str("id");
            if (index_.count(id)) e["id"].fail("duplicate entity id '" + id + "'");
            if (id.find(':') != std::string::npos) e["id"].fail("ids may not contain ':'");
            index_[id] = entities_.size();
            entities_.push_back(factory(e, *this));
        }
    for (auto& e : entities_) e->link(*this);

    for (const Spec& t : s.items("triggers")) {
        Trigger trig(t);
        Entity& target = entity(trig.action().target, t.path());
        const auto verbs = target.verbs();
        if (std::find(verbs.begin(), verbs.end(), trig.action().verb) == verbs.end()) {
            std::string list;
            for (const auto& v : verbs) list += (list.empty() ? "" : ", ") + v;
            t["do"].fail("'" + target.id() + "' (" + target.kind() + ") supports: " + (list.empty() ? "nothing" : list));
        }
        triggers_.push_back(std::move(trig));
    }
    for (const Spec& cs : s.items("cameras")) {
        auto cam = std::make_unique<Camera>(cs);
        for (const auto& ref : cam->referencedEntities()) entity(ref, cs.path());
        cameras_.push_back(std::move(cam));
    }
    for (const Spec& r : s.items("refine")) {
        Refinement ref;
        const auto between = r.strings("between");
        if (between.size() != 2) r["between"].fail("expected two entity ids");
        ref.a = between[0];
        ref.b = between[1];
        for (const auto& id : between)
            if (!entity(id, r.path()).isRigid()) r["between"].fail(id + " is not a rigid entity");
        ref.withinM = r.positive("within_m", 1.0);
        ref.holdS = r.positive("hold_s", 0.4);
        ref.factor = r.integer("factor", 8);
        if (ref.factor < 1 || ref.factor > 128) r["factor"].fail("must be in 1..128");
        refinements_.push_back(ref);
    }
    s.rejectUnknownKeys();
}

bool World::refining() {
    for (const Refinement& r : refinements_) {
        if (aabbGap(worldBounds(entity(r.a, "refine")), worldBounds(entity(r.b, "refine"))) < r.withinM)
            refineUntil_ = std::max(refineUntil_, t_ + r.holdS);
    }
    return t_ < refineUntil_;
}

void World::step(double h) {
    for (Trigger& trig : triggers_)
        if (auto a = trig.poll(t_, events_)) {
            events_.emit("trigger:" + a->target + ":" + a->verb, t_);
            entity(a->target, "trigger").act(*this, *a);
        }
    contacts_.beginStep(t_);
    for (auto& e : entities_) e->beforeStep(*this, h);
    scene_->simulate(float(h));
    scene_->fetchResults(true);
    scene_->fetchResultsParticleSystem();
    contacts_.endStep(h);
    t_ += h;
    ++steps_;
    for (auto& e : entities_) e->afterStep(*this, h);
}

void World::recordFrame(int frame, Recorder& rec) {
    rec.beginFrame(frame);
    Energy total;
    for (auto& e : entities_) {
        e->record(*this, rec);
        const Energy en = e->energy(*this);
        total.kinetic += en.kinetic;
        total.potential += en.potential;
    }
    const auto poseOf = [this](const std::string& id) { return entity(id, "camera").pose(); };
    for (auto& cam : cameras_) {
        const Pose p = cam->update(t_, poseOf);
        rec.put("camera/" + cam->id(), {float(p.p.x), float(p.p.y), float(p.p.z), float(p.q.x), float(p.q.y),
                                        float(p.q.z), float(p.q.w)});
    }
    rec.put("world/energy", {float(total.kinetic), float(total.potential), float(actuators_.netWork()),
                             float(actuators_.totalWorkIn())});
}

void World::run(const std::string& outDir, const Json& runInfo) {
    const auto t0 = Clk::now();
    Recorder rec;
    const double dt = clock_.dt();
    t_ = -clock_.prerollS;
    const long prerollSteps = std::lround(clock_.prerollS / dt);
    for (long i = 0; i < prerollSteps; ++i) step(dt);
    t_ = 0;  // remove rounding drift so frame times are exact
    int fine = 1;
    for (const auto& r : refinements_) fine = std::max(fine, r.factor);
    for (int f = 0; f < clock_.frames; ++f) {
        recordFrame(f, rec);
        if (f == clock_.frames - 1) break;
        for (int s = 0; s < clock_.substeps; ++s) {
            const int n = refining() ? fine : 1;
            fineSteps_ += n > 1 ? n : 0;
            for (int k = 0; k < n; ++k) step(dt / n);
        }
        t_ = (f + 1) / clock_.fps;
    }
    const double simSeconds = since(t0);

    std::filesystem::create_directories(outDir);
    Json x = describe();
    x["streams"] = rec.write(outDir);
    x["run"] = runInfo;
    x["run"]["wall_seconds"] = simSeconds;
    x["run"]["steps"] = steps_;
    x["run"]["fine_steps"] = fineSteps_;
    x["run"]["device"] = runtime_.deviceName();
    x["run"]["physx_messages"] = runtime_.errors().messages();
    x["run"]["physx_errors"] = runtime_.errors().errorCount();
    std::ofstream(outDir + "/x.json") << x.dump(1);
}

Json World::describe() const {
    Json ents = Json::array();
    for (const auto& e : entities_) {
        Json d = e->describe();
        d["id"] = e->id();
        d["kind"] = e->kind();
        ents.push_back(d);
    }
    Json cams = Json::array();
    for (const auto& c : cameras_) cams.push_back(c->toJson());
    Json trig = Json::array();
    for (const auto& t : triggers_) trig.push_back(t.describe());
    return {{"format", "p2s-x"},
            {"version", 1},
            {"scene", sceneInfo_},
            {"up", "z"},
            {"units", "SI (m, kg, s, N, W, J); angles in radians unless named _deg"},
            {"clock",
             {{"fps", clock_.fps}, {"frames", clock_.frames}, {"substeps", clock_.substeps}, {"dt", clock_.dt()},
              {"preroll_s", clock_.prerollS}}},
            {"gravity", toJson(gravity_)},
            {"wind", wind_.toJson()},
            {"materials", materials_.toJson()},
            {"entities", ents},
            {"cameras", cams},
            {"triggers", trig},
            {"events", events_.toJson()},
            {"contacts", contacts_.toJson()},
            {"actuators", actuators_.toJson()}};
}

}  // namespace p2s
