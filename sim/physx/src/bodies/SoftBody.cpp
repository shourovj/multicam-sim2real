#include "bodies/SoftBody.h"

#include <map>

#include "extensions/PxDeformableVolumeExt.h"
#include "extensions/PxRemeshingExt.h"

using namespace physx;

namespace p2s {

SoftBody::SoftBody(const Spec& s, World& w) : Entity(s.str("id"), "soft_body") {
    const auto shapes = Shape::parse(s["shape"], /*allowConcave=*/true);
    if (shapes.size() != 1) s["shape"].fail("needs exactly one closed shape (a container is several)");
    shape_ = shapes.front();
    hostId_ = s.str("host", "");
    pose_ = hostId_.empty() ? s.pose() : Pose{};
    if (s.has("mass_kg")) mass_ = s.positive("mass_kg");
    if (s.has("density")) density_ = s.positive("density");
    if (mass_ && density_) s.fail("give mass_kg or density, not both");
    youngs_ = s.positive("youngs_pa");
    poisson_ = s.num("poisson", 0.3);
    if (poisson_ < 0 || poisson_ >= 0.5) s["poisson"].fail("must be in [0, 0.5)");
    friction_ = s.num("friction", 0.5);
    damping_ = s.num("damping", 0.005);
    if (s.has("plastic")) {
        const Spec p = s["plastic"];
        plastic_ = {p.positive("yield_strain"), p.positive("low_limit", 0.2), p.positive("high_limit", 5.0)};
    }
    for (const Spec& a : s.items("attach")) {
        Aabb box;
        box.add(a.vec3("min"));
        box.add(a.vec3("max"));
        attach_.push_back(box);
    }
    resolution_ = s.integer("resolution", 20);
    if (resolution_ < 4 || resolution_ > 128) s["resolution"].fail("must be in 4..128");
    maxEdge_ = s.positive("max_edge_m", 0.12);
    iterations_ = s.integer("solver_iterations", 30);
    if (iterations_ < 1 || iterations_ > 255) s["solver_iterations"].fail("must be in 1..255");
    maxDepen_ = s.positive("max_depenetration_mps", 2.0);
    velocity0_ = s.vec3("velocity", {0, 0, 0});
    if (shape_.type == ShapeType::Convex) s["shape"].fail("use a prism, box or mesh: a raw point cloud has no surface yet");
}

void SoftBody::link(World& w) { build(w); }

void SoftBody::build(World& w) {
    PxPhysics& ph = w.physics();
    PxCudaContextManager& cuda = w.runtime().cuda();
    Entity* host = hostId_.empty() ? nullptr : &w.entity(hostId_, id());  // PhysX takes the host actor by pointer
    if (host && !host->isRigid()) throw SpecError(id() + ": host '" + hostId_ + "' is not a rigid body");
    const Pose hostPose = host ? host->pose() : Pose{};
    const Pose place = host ? hostPose * shape_.local : pose_ * shape_.local;

    // Closed surface in the body frame, remeshed so no edge is longer than max_edge_m, then voxel FEM.
    const TriMesh tm = shape_.surfaceMesh(24);
    PxArray<PxVec3> v;
    PxArray<PxU32> t;
    for (const Vec3& p : tm.vertices) v.pushBack(px(p));
    for (const auto& tri : tm.triangles)
        for (int k : tri) t.pushBack(PxU32(k));
    PxRemeshingExt::limitMaxEdgeLength(t, v, float(maxEdge_));
    PxCookingParams cp = w.runtime().cookingParams();
    cp.meshWeldTolerance = 0.001f;
    cp.meshPreprocessParams = PxMeshPreprocessingFlags(PxMeshPreprocessingFlag::eWELD_VERTICES);
    cp.buildTriangleAdjacencies = false;
    PxSimpleTriangleMesh sm;
    sm.points.count = v.size();
    sm.points.data = v.begin();
    sm.triangles.count = t.size() / 3;
    sm.triangles.data = t.begin();
    mesh_.reset(PxDeformableVolumeExt::createDeformableVolumeMesh(cp, sm, PxU32(resolution_),
                                                                  ph.getPhysicsInsertionCallback()));
    if (!mesh_) throw SpecError(id() + ": tetrahedral meshing failed (is the shape closed and not too thin?)");

    material_.reset(ph.createDeformableVolumeMaterial(float(youngs_), float(poisson_), float(friction_)));
    material_->setElasticityDamping(float(damping_));
    if (plastic_) {  // these three setters exist in the patched PhysX 5.9.1 build (third_party/*.patch)
        material_->setDeformThreshold(float((*plastic_)[0]));
        material_->setDeformLowLimitRatio(float((*plastic_)[1]));
        material_->setDeformHighLimitRatio(float((*plastic_)[2]));
    }
    body_.reset(ph.createDeformableVolume(cuda));
    if (!body_) throw std::runtime_error(id() + ": createDeformableVolume failed");
    PxDeformableVolumeMaterial* mats[] = {material_.get()};
    pxShape_.reset(ph.createShape(PxTetrahedronMeshGeometry(mesh_->getCollisionMesh()), mats, 1, true,
                                  PxShapeFlag::eSIMULATION_SHAPE));
    const CollisionFilter filter = w.filterFor(id(), host ? w.joinAssembly(*host) : "");
    pxShape_->setSimulationFilterData(filter.data());
    pxShape_->setContactOffset(0.01f);
    pxShape_->setRestOffset(0.0f);
    body_->attachShape(*pxShape_);
    body_->attachSimulationMesh(*mesh_->getSimulationMesh(), *mesh_->getDeformableVolumeAuxData());
    w.scene().addActor(*body_);

    PxVec4 *sp, *sv, *cpos, *rp;
    PxDeformableVolumeExt::allocateAndInitializeHostMirror(*body_, &cuda, sp, sv, cpos, rp);
    nSim_ = int(body_->getSimulationMesh()->getNbVertices());
    nColl_ = int(body_->getCollisionMesh()->getNbVertices());
    simPos_ = Pinned<PxVec4>::adopt(cuda, sp, size_t(nSim_));
    simVel_ = Pinned<PxVec4>::adopt(cuda, sv, size_t(nSim_));
    collPos_ = Pinned<PxVec4>::adopt(cuda, cpos, size_t(nColl_));
    const Pinned<PxVec4> restPos = Pinned<PxVec4>::adopt(cuda, rp, size_t(nColl_));
    PxDeformableVolumeExt::transform(*body_, px(place), 1.0f, sp, sv, cpos, rp);

    // Collision tetrahedra, their volume and the surface (faces used by exactly one tetrahedron).
    const PxTetrahedronMesh& cm = *body_->getCollisionMesh();
    const bool b16 = cm.getTetrahedronMeshFlags() & PxTetrahedronMeshFlag::e16_BIT_INDICES;
    for (PxU32 i = 0; i < cm.getNbTetrahedrons() * 4; ++i)
        tets_.push_back(int32_t(b16 ? static_cast<const PxU16*>(cm.getTetrahedrons())[i]
                                    : static_cast<const PxU32*>(cm.getTetrahedrons())[i]));
    const PxVec3* cv = cm.getVertices();
    std::map<std::array<int32_t, 3>, std::pair<std::array<int32_t, 3>, int>> faces;
    for (size_t k = 0; k < tets_.size(); k += 4) {
        const int32_t a = tets_[k], b = tets_[k + 1], c = tets_[k + 2], d = tets_[k + 3];
        volume_ += std::fabs((cv[b] - cv[a]).dot((cv[c] - cv[a]).cross(cv[d] - cv[a]))) / 6.0;
        const std::array<std::array<int32_t, 4>, 4> f = {{{a, b, c, d}, {a, b, d, c}, {a, c, d, b}, {b, c, d, a}}};
        for (const auto& [i, j, l, opposite] : f) {
            std::array<int32_t, 3> tri{i, j, l};
            // Wind each face away from the tetrahedron's fourth vertex, so surviving faces point outward.
            if ((cv[j] - cv[i]).cross(cv[l] - cv[i]).dot(cv[opposite] - cv[i]) > 0) std::swap(tri[1], tri[2]);
            std::array<int32_t, 3> key = tri;
            std::sort(key.begin(), key.end());
            auto& e = faces[key];
            e.first = tri;
            ++e.second;
        }
    }
    std::vector<char> onSurface(size_t(nColl_), 0);
    for (const auto& [_, f] : faces)
        if (f.second == 1) {
            triangles_.insert(triangles_.end(), f.first.begin(), f.first.end());
            for (int32_t i : f.first) onSurface[size_t(i)] = 1;
        }
    for (int i = 0; i < nColl_; ++i)
        if (onSurface[size_t(i)]) surfaceIndex_.push_back(i);

    const double mass = mass_ ? *mass_ : density_.value_or(w.materials().props(shape_.material).density) * volume_;
    PxDeformableVolumeExt::updateMass(*body_, float(mass / volume_), 50.0f, sp);
    const Vec3 vel = host ? host->velocity() : velocity0_;
    for (int i = 0; i < nSim_; ++i) simVel_[size_t(i)] = PxVec4(px(vel), 0.0f);
    PxDeformableVolumeExt::copyToDevice(*body_, PxDeformableVolumeDataFlag::eALL, sp, sv, cpos, rp);
    body_->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
    body_->setSolverIterationCounts(PxU32(iterations_));
    body_->setMaxDepenetrationVelocity(float(maxDepen_));
    body_->setSleepThreshold(0.0f);
    body_->setSettlingThreshold(0.0f);
    body_->setSettlingDamping(0.0f);
    body_->setWakeCounter(1e9f);

    // Bolt the simulation vertices inside the attach boxes to the host (host frame) or pin them to the world.
    std::vector<PxU32> idx;
    std::vector<PxVec4> coords;
    for (int i = 0; i < nSim_; ++i) {
        const Vec3 world = fromPx(simPos_[size_t(i)].getXYZ());
        const Vec3 local = host ? hostPose.transformInv(world) : pose_.transformInv(world);
        for (const Aabb& box : attach_)
            if (box.contains(local)) {
                idx.push_back(PxU32(i));
                coords.push_back(PxVec4(px(host ? local : world), 0.0f));
                break;
            }
    }
    nAttached_ = int(idx.size());
    if (!attach_.empty() && idx.empty())
        throw SpecError(id() + ": no vertex lies inside the attach boxes (they are in the " +
                        std::string(host ? "host's" : "body's") + " frame)");
    if (!idx.empty()) {
        PxDeformableAttachmentData ad;
        ad.actor[0] = body_.get();
        ad.type[0] = PxDeformableAttachmentTargetType::eVERTEX;
        ad.indices[0].data = idx.data();
        ad.indices[0].count = PxU32(idx.size());
        ad.actor[1] = host ? &host->actor() : nullptr;
        ad.type[1] = host ? PxDeformableAttachmentTargetType::eRIGID : PxDeformableAttachmentTargetType::eWORLD;
        ad.coords[1].data = coords.data();
        ad.coords[1].count = PxU32(coords.size());
        attachment_.reset(ph.createDeformableAttachment(ad));
        if (!attachment_) throw std::runtime_error(id() + ": createDeformableAttachment failed");
    }
    for (int i = 0; i < nColl_; ++i) rest_.push_back(fromPx(collPos_[size_t(i)].getXYZ()));
    if (host) for (Vec3& r : rest_) r = hostPose.transformInv(r);  // rest shape in the host frame
    current_.assign(rest_.size(), {});
    readBack(w, false);
}

void SoftBody::readBack(World& w, bool velocities) {
    PxCudaContextManager& cuda = w.runtime().cuda();
    PxScopedCudaLock lock(cuda);
    PxCudaContext* cc = cuda.getCudaContext();
    cc->memcpyDtoH(collPos_.data(), CUdeviceptr(body_->getPositionInvMassBufferD()), size_t(nColl_) * sizeof(PxVec4));
    for (int i = 0; i < nColl_; ++i) current_[size_t(i)] = fromPx(collPos_[size_t(i)].getXYZ());
    surface_.clear();
    for (int i : surfaceIndex_) surface_.push_back(current_[size_t(i)]);
    if (!velocities) return;
    cc->memcpyDtoH(simPos_.data(), CUdeviceptr(body_->getSimPositionInvMassBufferD()), size_t(nSim_) * sizeof(PxVec4));
    cc->memcpyDtoH(simVel_.data(), CUdeviceptr(body_->getSimVelocityBufferD()), size_t(nSim_) * sizeof(PxVec4));
    kinetic_ = potential_ = 0;
    for (int i = 0; i < nSim_; ++i) {
        const float invMass = simPos_[size_t(i)].w;
        if (invMass <= 0) continue;
        const double m = 1.0 / invMass;
        kinetic_ += 0.5 * m * simVel_[size_t(i)].getXYZ().magnitudeSquared();
        potential_ -= m * w.gravity().dot(fromPx(simPos_[size_t(i)].getXYZ()));
    }
}

void SoftBody::afterStep(World& w, double) { readBack(w, false); }

void SoftBody::record(World& w, Recorder& rec) {
    readBack(w, true);
    std::vector<float> verts;
    verts.reserve(current_.size() * 3);
    for (const Vec3& p : current_) verts.insert(verts.end(), {float(p.x), float(p.y), float(p.z)});
    rec.put("softbody/" + id() + "/verts", verts, 3);
    if (rec.frame() == 0) {
        rec.putStatic("softbody/" + id() + "/tets", std::vector<int32_t>(tets_), 4);
        rec.putStatic("softbody/" + id() + "/triangles", std::vector<int32_t>(triangles_), 3);
        std::vector<float> rest;
        for (const Vec3& p : rest_) rest.insert(rest.end(), {float(p.x), float(p.y), float(p.z)});
        rec.putStatic("softbody/" + id() + "/rest", rest, 3);
    }
}

Json SoftBody::describe() const {
    Json j = {{"host", hostId_},
              {"shape", shape_.toJson()},
              {"youngs_pa", youngs_},
              {"poisson", poisson_},
              {"plastic", plastic_.has_value()},
              {"volume_m3", volume_},
              {"vertices", nColl_},
              {"tetrahedra", tets_.size() / 4},
              {"attached_vertices", nAttached_},
              {"rest_frame", hostId_.empty() ? "world" : "host"},
              {"streams",
               {{"verts", "softbody/" + id() + "/verts"}, {"tets", "softbody/" + id() + "/tets"},
                {"triangles", "softbody/" + id() + "/triangles"}, {"rest", "softbody/" + id() + "/rest"}}}};
    if (plastic_) j["yield_strain"] = (*plastic_)[0];
    return j;
}

Energy SoftBody::energy(const World&) const { return {kinetic_, potential_}; }

}  // namespace p2s
