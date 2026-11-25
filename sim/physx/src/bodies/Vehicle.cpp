#include "bodies/Vehicle.h"

using namespace physx;

namespace p2s {

namespace {

constexpr uint32_t kWheelCategory = 4, kGroundCategory = 2;

}  // namespace

Vehicle::Vehicle(const Spec& s, World& w) : Entity(s.str("id"), "vehicle") {
    for (const Spec& sh : s.items("shapes"))
        for (Shape& x : Shape::parse(sh)) shapes_.push_back(std::move(x));
    if (shapes_.empty()) s.fail("needs chassis shapes");
    const Spec ws = s["wheels"];
    wheelRadius_ = ws.positive("radius_m");
    wheelHalfWidth_ = ws.positive("half_width_m", 0.11);
    const double wheelMass = ws.positive("mass_kg", 20);
    struct Axle {
        double x, track;
        bool steer, drive;
    };
    std::vector<Axle> axles;
    for (const Spec& a : ws.items("axles"))
        axles.push_back({a.num("x"), a.positive("track_m"), a.flag("steer", false), a.flag("drive", true)});
    if (axles.size() < 2) ws.fail("needs at least two axles");
    nWheels_ = int(axles.size() * 2);
    if (nWheels_ > int(PxVehicleLimits::eMAX_NB_WHEELS)) ws.fail("too many wheels");

    const Spec susp = s.has("suspension") ? s["suspension"] : Spec::fromJson(Json::object(), "suspension");
    const double travel = susp.positive("travel_m", 0.2), freq = susp.positive("frequency_hz", 1.6);
    const double zeta = susp.positive("damping_ratio", 0.3);
    const Spec lim = s["limits"];
    const double maxTorque = lim.positive("max_wheel_torque_nm"), maxBrake = lim.positive("max_brake_torque_nm");
    const double maxSteer = deg2rad(lim.positive("max_steer_deg", 30));
    const double mass = s.positive("mass_kg"), g = w.gravity().norm();

    Aabb box;
    for (const Shape& sh : shapes_) {
        const Aabb b = sh.bounds().transformed(sh.local);
        box.add(b.lo);
        box.add(b.hi);
    }
    const Vec3 dims = box.extent();
    const Vec3 com = s.vec3("com", {box.center().x, box.center().y, std::max(wheelRadius_, box.lo.z + 0.4 * dims.z)});

    // Vehicle frame = world-aligned body frame: +X forward, +Y left, +Z up.
    snippetvehicle::BaseVehicleParams& bp = vehicle_.mBaseParams;
    bp.axleDescription.setToDefault();
    for (size_t a = 0; a < axles.size(); ++a) {
        const PxU32 ids[2] = {PxU32(2 * a), PxU32(2 * a + 1)};
        bp.axleDescription.addAxle(2, ids);
    }
    bp.frame.lngAxis = PxVehicleAxes::ePosX;
    bp.frame.latAxis = PxVehicleAxes::ePosY;
    bp.frame.vrtAxis = PxVehicleAxes::ePosZ;
    bp.scale.scale = 1.0f;
    bp.rigidBodyParams.mass = float(mass);
    bp.rigidBodyParams.moi = PxVec3(float(mass / 12 * (dims.y * dims.y + dims.z * dims.z)),
                                    float(mass / 12 * (dims.x * dims.x + dims.z * dims.z)),
                                    float(mass / 12 * (dims.x * dims.x + dims.y * dims.y)));
    bp.suspensionStateCalculationParams.suspensionJounceCalculationType = PxVehicleSuspensionJounceCalculationType::eSWEEP;
    bp.suspensionStateCalculationParams.limitSuspensionExpansionVelocity = false;
    bp.brakeResponseParams[0].maxResponse = float(maxBrake);
    bp.brakeResponseParams[1].maxResponse = 0.0f;
    bp.steerResponseParams.maxResponse = float(maxSteer);
    PxVehicleDirectDriveThrottleCommandResponseParams& tp = vehicle_.mDirectDriveParams.directDriveThrottleResponseParams;
    tp.maxResponse = float(maxTorque);
    tp.nonlinearResponse.clear();

    // Suspension attachment: the wheel centre at full compression. At rest the spring sags g / w^2, so the
    // chassis origin settles on the ground.
    const double omega = 2 * kPi * freq, sag = g / (omega * omega);
    if (sag >= travel) susp["travel_m"].fail("must exceed the static sag g/(2 pi f)^2 = " + std::to_string(sag) + " m");
    std::vector<PxVec3> rel(static_cast<size_t>(nWheels_));
    for (int i = 0; i < nWheels_; ++i) {
        const Axle& a = axles[size_t(i / 2)];
        const Vec3 attach{a.x, (i % 2 == 0 ? 0.5 : -0.5) * a.track, wheelRadius_ + travel - sag};
        rel[size_t(i)] = px(attach - com);
    }
    std::vector<PxReal> sprung(static_cast<size_t>(nWheels_));
    PxVehicleComputeSprungMasses(PxU32(nWheels_), rel.data(), PxReal(mass), PxVehicleAxes::ePosZ, sprung.data());
    int nDriven = 0, nSteer = 0;
    for (int i = 0; i < nWheels_; ++i) {
        const Axle& a = axles[size_t(i / 2)];
        nDriven += a.drive;
        nSteer += a.steer;
        bp.brakeResponseParams[0].wheelResponseMultipliers[i] = 1.0f;
        bp.brakeResponseParams[1].wheelResponseMultipliers[i] = 0.0f;
        bp.steerResponseParams.wheelResponseMultipliers[i] = a.steer ? 1.0f : 0.0f;
        tp.wheelResponseMultipliers[i] = a.drive ? 1.0f : 0.0f;
        PxVehicleSuspensionParams& sp = bp.suspensionParams[i];
        sp.suspensionAttachment = PxTransform(rel[size_t(i)]);
        sp.suspensionTravelDir = PxVec3(0, 0, -1);
        sp.suspensionTravelDist = float(travel);
        sp.wheelAttachment = PxTransform(PxIdentity);
        PxVehicleSuspensionComplianceParams& sc = bp.suspensionComplianceParams[i];
        sc.wheelToeAngle.clear();
        sc.wheelToeAngle.addPair(0.0f, 0.0f);
        sc.wheelCamberAngle.clear();
        sc.wheelCamberAngle.addPair(0.0f, 0.0f);
        sc.suspForceAppPoint.clear();
        sc.suspForceAppPoint.addPair(0.0f, PxVec3(0, 0, 0));
        sc.tireForceAppPoint.clear();
        sc.tireForceAppPoint.addPair(0.0f, PxVec3(0, 0, 0));
        PxVehicleSuspensionForceParams& sf = bp.suspensionForceParams[i];
        sf.sprungMass = sprung[size_t(i)];
        sf.stiffness = float(sprung[size_t(i)] * omega * omega);
        sf.damping = float(zeta * 2 * std::sqrt(sf.stiffness * sprung[size_t(i)]));
        // Tyre: the SDK sample's stiffness ratios to rest load (as in the tank-truck crash).
        PxVehicleTireForceParams& tf = bp.tireForceParams[i];
        tf.restLoad = float(sprung[size_t(i)] * g);
        tf.longStiff = 4.36f * tf.restLoad;
        tf.latStiffX = 0.01f;
        tf.latStiffY = 21.1f * tf.restLoad;
        tf.camberStiff = 0.0f;
        const float fvs[3][2] = {{0.0f, 1.0f}, {0.1f, 1.0f}, {1.0f, 1.0f}};
        const float lf[2][2] = {{0.0f, 0.2308f}, {3.0f, 3.0f}};
        for (int k = 0; k < 3; ++k) tf.frictionVsSlip[k][0] = fvs[k][0], tf.frictionVsSlip[k][1] = fvs[k][1];
        for (int k = 0; k < 2; ++k) tf.loadFilter[k][0] = lf[k][0], tf.loadFilter[k][1] = lf[k][1];
        PxVehicleWheelParams& wp = bp.wheelParams[i];
        wp.radius = float(wheelRadius_);
        wp.halfWidth = float(wheelHalfWidth_);
        wp.mass = float(wheelMass);
        wp.moi = float(0.5 * wheelMass * wheelRadius_ * wheelRadius_);
        wp.dampingRate = 0.25f;
    }
    if (nDriven == 0) ws.fail("no axle has drive: true");
    if (nSteer > 0) {
        bp.ackermannParams[0].wheelIds[0] = 0;
        bp.ackermannParams[0].wheelIds[1] = 1;
        bp.ackermannParams[0].wheelBase = float(axles.front().x - axles.back().x);
        bp.ackermannParams[0].trackWidth = float(axles.front().track);
        bp.ackermannParams[0].strength = 1.0f;
    }
    limits_ = {mass, maxTorque * nDriven / wheelRadius_, maxBrake * nWheels_ / wheelRadius_,
               axles.front().x - axles.back().x, maxSteer, dims.x};

    friction_.material = &w.materials().get("asphalt", id());
    friction_.friction = float(s.positive("tyre_friction", 0.9));
    vehicle_.mPhysXParams.create(bp.axleDescription, PxQueryFilterData(PxFilterData(), PxQueryFlag::eSTATIC), nullptr,
                                 &friction_, 1, friction_.friction, px(Pose{com, {}}), PxVec3(0.3f), PxTransform(PxIdentity));
    if (!vehicle_.initialize(w.physics(), w.runtime().cookingParams(), w.materials().get("tyre", id())))
        throw SpecError(id() + ": the Vehicle SDK rejected these parameters");
    vehicle_.mTransmissionCommandState.gear = PxVehicleDirectDriveTransmissionCommandState::eFORWARD;
    const Pose start = s.pose();
    vehicle_.setUpActor(w.scene(), px(start), id().c_str());

    const std::string assembly = s.str("assembly", id());
    w.shapeFactory().attach(chassis(), shapes_, w.filterFor(id(), assembly), id());
    CollisionFilter wheelFilter = w.filterFor(id(), assembly, kWheelCategory, kGroundCategory);
    for (int i = 0; i < nWheels_; ++i) {  // wheels hit other bodies, never the road: the suspension carries the car
        PxShape* wsh = vehicle_.mPhysXState.physxActor.wheelShapes[i];
        wsh->setFlag(PxShapeFlag::eSIMULATION_SHAPE, true);
        wsh->setSimulationFilterData(wheelFilter.data());
    }
    chassis().setSolverIterationCounts(16, 4);
    chassis().setMaxDepenetrationVelocity(float(s.positive("max_depenetration_mps", 2.0)));
    const double v0 = s.num("speed_mps", 0);
    chassis().setLinearVelocity(px(start.q.axisX() * v0));
    for (int i = 0; i < nWheels_; ++i) vehicle_.mBaseState.wheelRigidBody1dStates[i].rotationSpeed = float(v0 / wheelRadius_);
    lastVel_ = velocity();

    ctx_.setToDefault();
    ctx_.frame = bp.frame;
    ctx_.scale = bp.scale;
    ctx_.gravity = px(w.gravity());
    ctx_.physxScene = &w.scene();
    ctx_.physxActorUpdateMode = PxVehiclePhysXActorUpdateMode::eAPPLY_ACCELERATION;

    if (s.has("driver")) driver_.emplace(s["driver"]);
    w.actuators().declare(id() + "/drive", "wheel_drive", mass, true);
    w.actuators().declare(id() + "/brakes", "brake", mass, true);
}

Vehicle::~Vehicle() { vehicle_.destroy(); }

PxRigidDynamic& Vehicle::chassis() const { return *vehicle_.mPhysXState.physxActor.rigidBody->is<PxRigidDynamic>(); }

Pose Vehicle::pose() const { return fromPx(chassis().getGlobalPose()); }

Vec3 Vehicle::velocity() const { return fromPx(chassis().getLinearVelocity()); }

Pose Vehicle::wheelPose(int i) const {
    return pose() * fromPx(vehicle_.mPhysXState.physxActor.wheelShapes[i]->getLocalPose());
}

void Vehicle::link(World& w) {
    if (driver_) driver_->link(w, id());
}

void Vehicle::beforeStep(World& w, double dt) {
    PxVehicleCommandState& cmd = vehicle_.mCommandState;
    cmd.setToDefault();
    cmd.nbBrakes = 1;
    const double speed = velocity().dot(pose().q.axisX());
    DriveCommand dc;
    if (driver_) dc = driver_->update(w, id(), pose(), speed, limits_);
    if (released_) dc = {};
    if (manualBrake_ >= 0) dc = {0, manualBrake_, dc.steer};
    cmd.throttle = float(dc.throttle);
    cmd.brakes[0] = float(dc.brake);
    cmd.steer = float(dc.steer);
    lastCmd_ = dc;
    vehicle_.step(float(dt), ctx_);
    logActuators(w, dt);
}

void Vehicle::logActuators(World& w, double dt) {
    double drivePower = 0, driveTorque = 0, brakePower = 0, brakeTorque = 0;
    for (int i = 0; i < nWheels_; ++i) {
        const double omega = vehicle_.mBaseState.wheelRigidBody1dStates[i].rotationSpeed;
        const double tq = vehicle_.mDirectDriveState.directDriveThrottleResponseStates[i];
        const double br = vehicle_.mBaseState.brakeCommandResponseStates[i];
        driveTorque += tq;
        drivePower += tq * omega;
        brakeTorque += br;
        brakePower += br * std::fabs(omega);
    }
    w.actuators().step(id() + "/drive", driveTorque, drivePower, dt);
    w.actuators().step(id() + "/brakes", brakeTorque, -brakePower, dt);
}

void Vehicle::afterStep(World& w, double dt) {
    const Vec3 v = velocity();
    peakAccel_ = std::max(peakAccel_, ((v - lastVel_) / dt - w.gravity()).norm());
    peakSpeed_ = std::max(peakSpeed_, v.norm());
    lastVel_ = v;
}

void Vehicle::record(World&, Recorder& rec) {
    const Pose p = pose();
    rec.put("body/" + id() + "/pose",
            {float(p.p.x), float(p.p.y), float(p.p.z), float(p.q.x), float(p.q.y), float(p.q.z), float(p.q.w)});
    const PxVec3 v = chassis().getLinearVelocity(), om = chassis().getAngularVelocity();
    rec.put("body/" + id() + "/vel", {v.x, v.y, v.z, om.x, om.y, om.z});
    std::vector<float> wheels;
    for (int i = 0; i < nWheels_; ++i) {
        const Pose wp = wheelPose(i);
        wheels.insert(wheels.end(), {float(wp.p.x), float(wp.p.y), float(wp.p.z), float(wp.q.x), float(wp.q.y),
                                     float(wp.q.z), float(wp.q.w)});
    }
    rec.put("vehicle/" + id() + "/wheels", wheels, 7);
    rec.put("vehicle/" + id() + "/controls",
            {float(lastCmd_.throttle), float(lastCmd_.brake), float(lastCmd_.steer), float(v.dot(px(p.q.axisX())))});
}

Json Vehicle::describe() const {
    Json chassisShapes = Json::array();
    for (const Shape& s : shapes_) chassisShapes.push_back(s.toJson());
    Json parts = {{{"name", "chassis"}, {"shapes", chassisShapes}, {"stream", "body/" + id() + "/pose"}, {"row", 0}}};
    // The PhysX wheel shape's axis is the vehicle's lateral axis; p2s cylinders run along local Z.
    const Shape wheel = Shape::cylinder(wheelRadius_, wheelHalfWidth_, Pose{{}, Quat::axisAngle({1, 0, 0}, kPi / 2)});
    for (int i = 0; i < nWheels_; ++i)
        parts.push_back({{"name", "wheel" + std::to_string(i)}, {"shapes", {wheel.toJson()}},
                         {"stream", "vehicle/" + id() + "/wheels"}, {"row", i}, {"kind", "wheel"}});
    Json j = {{"parts", parts},
              {"mass_kg", limits_.massKg},
              {"wheel_radius_m", wheelRadius_},
              {"peak_accel_mps2", peakAccel_},
              {"peak_speed_mps", peakSpeed_},
              {"streams",
               {{"pose", "body/" + id() + "/pose"}, {"vel", "body/" + id() + "/vel"},
                {"wheels", "vehicle/" + id() + "/wheels"}, {"controls", "vehicle/" + id() + "/controls"}}},
              {"stream_columns", {{"controls", {"throttle", "brake", "steer", "forward_speed_mps"}}}}};
    if (driver_) j["driver"] = driver_->toJson();
    return j;
}

void Vehicle::act(World& w, const Action& a) {
    const Spec p = Spec::fromJson(a.params, id() + "." + a.verb);
    if (a.verb == "brake") {
        manualBrake_ = std::clamp(p.num("level", 1.0), 0.0, 1.0);
    } else if (a.verb == "release") {
        released_ = true;
    } else if (a.verb == "set_speed") {
        if (!driver_) throw SpecError(id() + ": set_speed needs a driver");
        driver_->setCruise(p.positive("mps"));
    }
    w.events().emit(id() + ":" + a.verb, w.time());
}

Energy Vehicle::energy(const World& w) const {
    const PxRigidDynamic& d = chassis();
    const double m = d.getMass();
    const PxTransform com = d.getGlobalPose() * d.getCMassLocalPose();
    const PxVec3 om = com.q.rotateInv(d.getAngularVelocity());
    const PxVec3 I = d.getMassSpaceInertiaTensor();
    const double rot = 0.5 * (I.x * om.x * om.x + I.y * om.y * om.y + I.z * om.z * om.z);
    double wheels = 0;
    for (int i = 0; i < nWheels_; ++i) {
        const double omega = vehicle_.mBaseState.wheelRigidBody1dStates[i].rotationSpeed;
        wheels += 0.5 * vehicle_.mBaseParams.wheelParams[i].moi * omega * omega;
    }
    return {0.5 * m * fromPx(d.getLinearVelocity()).norm2() + rot + wheels, -m * w.gravity().dot(fromPx(com.p))};
}

}  // namespace p2s
