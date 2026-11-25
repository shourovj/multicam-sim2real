#include "bodies/Cloth.h"

#include "extensions/PxDeformableSurfaceExt.h"

using namespace physx;

namespace p2s {

Cloth::Cloth(const Spec& s, World&) : Entity(s.str("id"), "cloth") {
    const auto size = s.numbers("size_m");
    if (size.size() != 2 || size[0] <= 0 || size[1] <= 0) s["size_m"].fail("expected [width, height] > 0");
    width_ = size[0];
    height_ = size[1];
    if (s.has("grid")) {
        const auto g = s.numbers("grid");
        if (g.size() != 2 || g[0] < 2 || g[1] < 2 || g[0] * g[1] > 200000) s["grid"].fail("expected [nu, nv], 2..~400 each");
        nu_ = int(g[0]);
        nv_ = int(g[1]);
    } else {  // about 6 cm cells
        nu_ = std::max(2, int(width_ / 0.06) + 1);
        nv_ = std::max(2, int(height_ / 0.06) + 1);
    }
    pose_ = s.pose();
    hostId_ = s.str("host", "");
    for (const Spec& p : s.items("pin")) {
        const auto lo = p.numbers("min"), hi = p.numbers("max");
        if (lo.size() != 2 || hi.size() != 2) p.fail("pin boxes are [x, y] in the cloth's plane");
        Aabb box;
        box.add({lo[0] - 1e-4, lo[1] - 1e-4, -1});
        box.add({hi[0] + 1e-4, hi[1] + 1e-4, 1});
        pins_.push_back(box);
    }
    areal_ = s.positive("areal_density", 0.25);
    thickness_ = s.positive("thickness_m", 0.002);
    youngs_ = s.positive("youngs_pa", 5e6);
    poisson_ = s.num("poisson", 0.3);
    bending_ = s.num("bending", 2e-6);
    friction_ = s.num("friction", 0.5);
    damping_ = s.num("damping", 0.02);
    if (s.has("drag")) {
        cn_ = s["drag"].num("normal", 1.2);
        ct_ = s["drag"].num("tangent", 0.15);
    }
    iterations_ = s.integer("solver_iterations", 30);
}

void Cloth::link(World& w) {
    PxPhysics& ph = w.physics();
    PxCudaContextManager& cuda = w.runtime().cuda();
    std::vector<PxVec3> verts;
    std::vector<PxU32> tris;
    for (int i = 0; i < nu_; ++i)
        for (int j = 0; j < nv_; ++j) verts.push_back(PxVec3(float(width_ * i / (nu_ - 1)), float(height_ * j / (nv_ - 1)), 0));
    for (int i = 1; i < nu_; ++i)
        for (int j = 1; j < nv_; ++j) {
            const PxU32 a = PxU32((i - 1) * nv_ + j - 1), b = PxU32(i * nv_ + j - 1), c = PxU32((i - 1) * nv_ + j),
                        d = PxU32(i * nv_ + j);
            tris.insert(tris.end(), {a, b, c, c, b, d});
        }
    PxCookingParams cp = w.runtime().cookingParams();
    cp.meshWeldTolerance = 0.001f;
    cp.meshPreprocessParams = PxMeshPreprocessingFlags(PxMeshPreprocessingFlag::eWELD_VERTICES) |
                              PxMeshPreprocessingFlag::eENABLE_VERT_MAPPING;
    cp.buildTriangleAdjacencies = false;
    cp.midphaseDesc = PxMeshMidPhase::eBVH34;
    PxTriangleMeshDesc md;
    md.points.count = PxU32(verts.size());
    md.points.stride = sizeof(PxVec3);
    md.points.data = verts.data();
    md.triangles.count = PxU32(tris.size() / 3);
    md.triangles.stride = 3 * sizeof(PxU32);
    md.triangles.data = tris.data();
    mesh_.reset(PxCreateTriangleMesh(cp, md, ph.getPhysicsInsertionCallback()));
    if (!mesh_) throw SpecError(id() + ": cloth mesh cooking failed");

    material_.reset(ph.createDeformableSurfaceMaterial(float(youngs_), float(poisson_), float(friction_),
                                                       float(thickness_), float(bending_)));
    surface_.reset(ph.createDeformableSurface(cuda));
    if (!surface_) throw std::runtime_error(id() + ": createDeformableSurface failed");
    PxDeformableSurfaceMaterial* mats[] = {material_.get()};
    pxShape_.reset(ph.createShape(PxTriangleMeshGeometry(mesh_.get()), mats, 1, true,
                                  PxShapeFlag::eVISUALIZATION | PxShapeFlag::eSIMULATION_SHAPE));
    pxShape_->setSimulationFilterData(w.filterFor(id(), hostId_).data());
    surface_->attachShape(*pxShape_);
    w.scene().addActor(*surface_);

    std::optional<std::reference_wrapper<Entity>> host;
    if (!hostId_.empty()) host = w.entity(hostId_, id());
    const Pose world = host ? host->get().pose() * pose_ : pose_;
    PxVec4 *p, *v, *r;
    nVerts_ = int(PxDeformableSurfaceExt::allocateAndInitializeHostMirror(*surface_, verts.data(), nullptr, verts.data(),
                                                                          0.5f, px(world), &cuda, p, v, r));
    pos_ = Pinned<PxVec4>::adopt(cuda, p, size_t(nVerts_));
    vel_ = Pinned<PxVec4>::adopt(cuda, v, size_t(nVerts_));
    rest_ = Pinned<PxVec4>::adopt(cuda, r, size_t(nVerts_));
    // The cooked mesh may reorder vertices and triangles; take both from it.
    const PxU32 ntri = mesh_->getNbTriangles();
    const bool b16 = mesh_->getTriangleMeshFlags() & PxTriangleMeshFlag::e16_BIT_INDICES;
    std::vector<PxReal> triMass;
    for (PxU32 t = 0; t < ntri; ++t) {
        std::array<int32_t, 3> tri;
        for (int k = 0; k < 3; ++k)
            tri[size_t(k)] = int32_t(b16 ? static_cast<const PxU16*>(mesh_->getTriangles())[3 * t + PxU32(k)]
                                         : static_cast<const PxU32*>(mesh_->getTriangles())[3 * t + PxU32(k)]);
        triangles_.push_back(tri);
        const PxVec3 a = mesh_->getVertices()[tri[0]], b = mesh_->getVertices()[tri[1]], c = mesh_->getVertices()[tri[2]];
        triMass.push_back(float(0.5 * (b - a).cross(c - a).magnitude() * areal_));
    }
    PxDeformableSurfaceExt::distributeTriangleMassToVertices(*surface_, triMass.data(), p);
    pxShape_->setContactOffset(float(2 * thickness_));
    pxShape_->setRestOffset(float(thickness_));
    surface_->setLinearDamping(float(damping_));
    surface_->setSleepThreshold(0.0f);
    surface_->setSettlingThreshold(0.0f);
    surface_->setSettlingDamping(0.0f);
    surface_->setWakeCounter(1e9f);
    surface_->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
    surface_->setMaxLinearVelocity(100.0f);
    surface_->setSolverIterationCounts(PxU32(iterations_), 1);
    PxDeformableSurfaceExt::copyToDevice(*surface_, PxDeformableSurfaceDataFlag::eALL, PxU32(nVerts_), p, v, r);

    std::vector<PxU32> pinIdx;
    std::vector<PxVec4> pinPos;
    const Pose hostPose = host ? host->get().pose() : Pose{};
    for (int i = 0; i < nVerts_; ++i) {
        const Vec3 wp = fromPx(pos_[size_t(i)].getXYZ());
        const Vec3 local = world.transformInv(wp);
        uv_.insert(uv_.end(), {float(local.x / width_), float(local.y / height_)});
        for (const Aabb& box : pins_)
            if (box.contains(local)) {
                pinIdx.push_back(PxU32(i));
                pinPos.push_back(PxVec4(px(host ? hostPose.transformInv(wp) : wp), 0.0f));
                break;
            }
    }
    nPinned_ = int(pinIdx.size());
    if (!pins_.empty() && pinIdx.empty()) throw SpecError(id() + ": no vertex lies inside the pin boxes");
    if (!pinIdx.empty()) {
        PxDeformableAttachmentData ad;
        ad.actor[0] = surface_.get();
        ad.type[0] = PxDeformableAttachmentTargetType::eVERTEX;
        ad.indices[0].data = pinIdx.data();
        ad.indices[0].count = PxU32(pinIdx.size());
        ad.actor[1] = host ? &host->get().actor() : nullptr;
        ad.type[1] = host ? PxDeformableAttachmentTargetType::eRIGID : PxDeformableAttachmentTargetType::eWORLD;
        ad.coords[1].data = pinPos.data();
        ad.coords[1].count = PxU32(pinPos.size());
        attachment_.reset(ph.createDeformableAttachment(ad));
        if (!attachment_) throw std::runtime_error(id() + ": createDeformableAttachment failed");
    }
    current_.resize(size_t(nVerts_));
    readBack(w);
}

void Cloth::readBack(World& w) {
    PxCudaContextManager& cuda = w.runtime().cuda();
    PxScopedCudaLock lock(cuda);
    PxCudaContext* cc = cuda.getCudaContext();
    cc->memcpyDtoH(pos_.data(), CUdeviceptr(surface_->getPositionInvMassBufferD()), size_t(nVerts_) * sizeof(PxVec4));
    cc->memcpyDtoH(vel_.data(), CUdeviceptr(surface_->getVelocityBufferD()), size_t(nVerts_) * sizeof(PxVec4));
    kinetic_ = potential_ = 0;
    for (int i = 0; i < nVerts_; ++i) {
        current_[size_t(i)] = fromPx(pos_[size_t(i)].getXYZ());
        const float invMass = pos_[size_t(i)].w;
        if (invMass <= 0) continue;
        kinetic_ += 0.5 / invMass * vel_[size_t(i)].getXYZ().magnitudeSquared();
        potential_ -= w.gravity().dot(current_[size_t(i)]) / invMass;
    }
}

void Cloth::applyWind(World& w, double dt) {
    // Quadratic drag on each triangle's normal relative wind holds a flag out; a weaker tangential term
    // models the flutter-averaged skin drag. Applied as velocity changes split over the three vertices.
    const double rho = w.wind().airDensity();
    for (const auto& tri : triangles_) {
        const Vec3 &a = current_[size_t(tri[0])], &b = current_[size_t(tri[1])], &c = current_[size_t(tri[2])];
        Vec3 n = (b - a).cross(c - a);
        const double area = 0.5 * n.norm();
        if (area < 1e-9) continue;
        n = n.normalized();
        Vec3 vt;
        for (int32_t k : tri) vt += fromPx(vel_[size_t(k)].getXYZ()) / 3.0;
        const Vec3 rel = w.wind().at(w.time(), (a + b + c) / 3.0) - vt;
        const double vn = rel.dot(n);
        const Vec3 tang = rel - n * vn;
        const Vec3 f = n * (0.5 * rho * cn_ * area * vn * std::fabs(vn)) + tang * (0.5 * rho * ct_ * area * tang.norm());
        for (int32_t k : tri) vel_[size_t(k)] += PxVec4(px(f * (pos_[size_t(k)].w * dt / 3.0)), 0.0f);  // w = 1/m
    }
    PxDeformableSurfaceExt::copyToDevice(*surface_, PxDeformableSurfaceDataFlag::eVELOCITY, PxU32(nVerts_), pos_.data(),
                                         vel_.data(), rest_.data());
}

void Cloth::beforeStep(World& w, double dt) {
    if (w.wind().mean().norm2() > 0) applyWind(w, dt);
}

void Cloth::afterStep(World& w, double) { readBack(w); }

void Cloth::record(World& w, Recorder& rec) {
    readBack(w);
    std::vector<float> v;
    v.reserve(current_.size() * 3);
    for (const Vec3& p : current_) v.insert(v.end(), {float(p.x), float(p.y), float(p.z)});
    rec.put("cloth/" + id() + "/verts", v, 3);
    if (rec.frame() == 0) {
        std::vector<int32_t> t;
        for (const auto& tri : triangles_) t.insert(t.end(), tri.begin(), tri.end());
        rec.putStatic("cloth/" + id() + "/triangles", t, 3);
        rec.putStatic("cloth/" + id() + "/uv", uv_, 2);
    }
}

Json Cloth::describe() const {
    return {{"size_m", {width_, height_}},
            {"vertices", nVerts_},
            {"pinned_vertices", nPinned_},
            {"host", hostId_},
            {"areal_density", areal_},
            {"streams",
             {{"verts", "cloth/" + id() + "/verts"}, {"triangles", "cloth/" + id() + "/triangles"},
              {"uv", "cloth/" + id() + "/uv"}}}};
}

Energy Cloth::energy(const World&) const { return {kinetic_, potential_}; }

}  // namespace p2s
