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
        const Aabb b = s.bounds().transformed(p * s.local);
        box.add(b.lo);
        box.add(b.hi);
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
    if (!assembly.empty()) {
        f.assemblyId = assemblies_.try_emplace(assembly, uint32_t(assemblies_.size() + 1)).first->second;
        assemblyOf_[entityId] = assembly;
    }
    return f;
}

void World::build(const Spec& s) {
    if (s.integer("p2s_version", 1) != 1) s["p2s_version"].fail("only version 1 is supported");
    sceneInfo_ = {{"name", s.str("name")}};
    for (const char* passthrough : {"meta", "expect", "acknowledge"})  // read by the checker, not the simulator
        if (s.has(passthrough)) (void)s[passthrough].opaque();
    if (s.has("meta")) sceneInfo_["meta"] = s["meta"].opaque();

    const Spec c = s["clock"];
    clock_.fps = c.positive("fps", 24);
    const double duration = c.positive("duration_s");
    if (duration * clock_.fps > 1e5) c["duration_s"].fail("more than 100000 frames; shorten the clip or lower fps");
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

    // world.ground: absent or true = default ground, an object = configured ground, false = none.
    if (w.has("ground") && !w["ground"].isObject() && !w["ground"].raw().is_boolean())
        w["ground"].fail("expected true, false or an object {material, height}");
    if (!w.has("ground") || w["ground"].isObject() || w.flag("ground", true)) {
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
    detectDeformableContacts();
}

void World::detectDeformableContacts() {
    constexpr double kTouch = 0.015;     // m: closer than this counts as touching (PhysX contact offsets are ~1 cm)
    constexpr size_t kMaxPoints = 4000;  // points tested per deformable per step
    const auto sameAssembly = [this](const std::string& a, const std::string& b) {
        const auto ia = assemblyOf_.find(a), ib = assemblyOf_.find(b);
        return ia != assemblyOf_.end() && ib != assemblyOf_.end() && ia->second == ib->second;
    };
    for (size_t ei = 0; ei < entities_.size(); ++ei) {
        const Entity* e = entities_[ei].get();
        const std::span<const Vec3> pts = e->contactPoints();
        if (pts.empty()) continue;
        const size_t stride = std::max<size_t>(1, pts.size() / kMaxPoints);
        Aabb box;
        for (const Vec3& p : pts) box.add(p);
        for (size_t oi = 0; oi < entities_.size(); ++oi) {
            const Entity* o = entities_[oi].get();
            if (oi == ei || sameAssembly(e->id(), o->id()) || o->id() == e->contactAlias()) continue;
            const std::span<const Vec3> other = o->contactPoints();
            bool touching = false;
            if (!other.empty()) {
                if (oi < ei) continue;  // each deformable pair once
                Aabb ob;
                for (const Vec3& p : other) ob.add(p);
                if (aabbGap(box, ob) > kTouch) continue;
                // Spatial hash of the other body's points, cell = touch distance.
                std::map<std::array<long, 3>, std::vector<Vec3>> grid;
                const auto cell = [](const Vec3& p) {
                    return std::array<long, 3>{long(std::floor(p.x / kTouch)), long(std::floor(p.y / kTouch)),
                                               long(std::floor(p.z / kTouch))};
                };
                for (size_t i = 0; i < other.size(); i += std::max<size_t>(1, other.size() / kMaxPoints))
                    grid[cell(other[i])].push_back(other[i]);
                for (size_t i = 0; i < pts.size() && !touching; i += stride) {
                    const auto c = cell(pts[i]);
                    for (long dx = -1; dx <= 1 && !touching; ++dx)
                        for (long dy = -1; dy <= 1 && !touching; ++dy)
                            for (long dz = -1; dz <= 1 && !touching; ++dz) {
                                const auto it = grid.find({c[0] + dx, c[1] + dy, c[2] + dz});
                                if (it == grid.end()) continue;
                                for (const Vec3& q : it->second)
                                    if ((q - pts[i]).norm() < kTouch) touching = true;
                            }
                }
            } else if (!o->shapes().empty()) {
                if (aabbGap(box, worldBounds(*o)) > kTouch) continue;
                const PosedShapes body{o->shapes(), o->pose()};
                for (size_t i = 0; i < pts.size() && !touching; i += stride) touching = body.sdf(pts[i]) < kTouch;
            }
            if (!touching) continue;
            contacts_.touch(e->id(), o->id(), t_);
            const std::string a = e->contactAlias().empty() ? e->id() : e->contactAlias();
            const std::string b = o->contactAlias().empty() ? o->id() : o->contactAlias();
            if (a != e->id() || b != o->id()) contacts_.touch(a, b, t_);
        }
    }
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
    std::ofstream out(outDir + "/x.json");
    out << x.dump(1);
    if (!out) throw std::runtime_error("failed writing " + outDir + "/x.json");
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
