#include "control/Driver.h"

#include "bodies/Signal.h"
#include "bodies/Walker.h"

namespace p2s {

Polyline::Polyline(std::vector<std::array<double, 2>> pts) : p_(std::move(pts)) {
    s_ = {0};
    for (size_t i = 1; i < p_.size(); ++i) s_.push_back(s_.back() + std::hypot(p_[i][0] - p_[i - 1][0], p_[i][1] - p_[i - 1][1]));
}

Polyline Polyline::parse(const Spec& list) {
    if (!list.isArray()) list.fail("expected a list of [x, y] points");
    std::vector<std::array<double, 2>> pts;
    for (size_t i = 0; i < list.size(); ++i) {
        const Spec xy = list.at(i);
        if (!xy.isArray() || xy.size() != 2) xy.fail("expected [x, y]");
        pts.push_back({xy.at(0).asNum(), xy.at(1).asNum()});
    }
    if (pts.size() < 2) list.fail("needs at least two points");
    Polyline line(std::move(pts));
    if (line.length() < 1e-6) list.fail("has zero length");
    return line;
}

Vec3 Polyline::at(double s) const {
    s = std::clamp(s, 0.0, length());
    size_t i = 0;
    while (i + 2 < s_.size() && s_[i + 1] < s) ++i;
    const double u = (s - s_[i]) / std::max(s_[i + 1] - s_[i], 1e-9);
    return {p_[i][0] + u * (p_[i + 1][0] - p_[i][0]), p_[i][1] + u * (p_[i + 1][1] - p_[i][1]), 0};
}

Vec3 Polyline::tangent(double s) const {
    // A 1 m chord smooths the corners of the polyline; it is kept inside the path so the ends have a direction.
    const double h = 0.5, a = std::clamp(s - h, 0.0, std::max(0.0, length() - 2 * h));
    return (at(a + 2 * h) - at(a)).normalized();
}

double Polyline::project(const Vec3& p) const {
    double best = 1e300, bestS = 0;
    for (size_t i = 0; i + 1 < p_.size(); ++i) {
        const Vec3 a{p_[i][0], p_[i][1], 0}, b{p_[i + 1][0], p_[i + 1][1], 0};
        const Vec3 ab = b - a, q{p.x, p.y, 0};
        const double u = std::clamp((q - a).dot(ab) / std::max(ab.norm2(), 1e-12), 0.0, 1.0);
        const double d = (q - (a + ab * u)).norm2();
        if (d < best) {
            best = d;
            bestS = s_[i] + u * (s_[i + 1] - s_[i]);
        }
    }
    return bestS;
}

double Polyline::lateral(const Vec3& p) const {
    const double s = project(p);
    const Vec3 t = tangent(s), d = Vec3{p.x, p.y, 0} - at(s);
    return t.cross(d).z;
}

std::vector<double> Polyline::speedLimit(double cruise, double latAccel, double brake) const {
    // Sampled every metre: curvature from the change of heading, then a backward pass so the vehicle starts
    // braking for a bend early enough to take it at the bend limit.
    const int n = std::max(2, int(std::ceil(length())) + 1);
    std::vector<double> v(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double s = std::min(double(i), length());
        const double s0 = std::max(0.0, s - 2.0), s1 = std::min(length(), s + 2.0);
        const double kappa = std::acos(std::clamp(tangent(s0).dot(tangent(s1)), -1.0, 1.0)) / std::max(s1 - s0, 1.0);
        v[size_t(i)] = std::min(cruise, std::sqrt(latAccel / std::max(kappa, 1e-6)));
    }
    for (int i = n - 2; i >= 0; --i) v[size_t(i)] = std::min(v[size_t(i)], std::sqrt(v[size_t(i + 1)] * v[size_t(i + 1)] + 2 * brake));
    return v;
}

Driver::Driver(const Spec& s) {
    path_ = Polyline::parse(s["path"]);
    cruise_ = s.positive("cruise_mps", 10);
    accelMax_ = s.positive("accel_max", 2.0);
    brakeComfort_ = s.positive("brake_comfort", 2.0);
    brakeMax_ = s.positive("brake_max", 6.0);
    brakeYellow_ = s.positive("brake_yellow", 3.0);
    latAccel_ = s.positive("lat_accel_max", 2.5);
    headway_ = s.positive("headway_s", 1.2);
    gapMin_ = s.positive("gap_min_m", 2.0);
    laneWidth_ = s.positive("lane_width_m", 3.5);
    leaderId_ = s.str("leader", "");
    signalId_ = s.str("signal", "");
    if (s.has("stop_s")) stopS_ = s.num("stop_s");
    if (!signalId_.empty() && !stopS_) s.fail("a signal needs stop_s (arc length of the stop line)");
    runsRed_ = s.flag("runs_red", false);
    if (runsRed_ && signalId_.empty()) s["runs_red"].fail("runs_red needs the signal it runs (and its stop_s)");
    if (s.has("yield_to")) yieldIds_ = s.strings("yield_to");
    for (const Spec& c : s.items("crosswalks")) {
        if (!c.isArray() || c.size() != 2) c.fail("expected [s0, s1]");
        crosswalks_.push_back({c.at(0).asNum(), c.at(1).asNum()});
        if (crosswalks_.back()[1] <= crosswalks_.back()[0]) c.fail("expected [s0, s1] with s1 > s0");
    }
    if (!yieldIds_.empty() && crosswalks_.empty()) s.fail("yield_to needs crosswalks");
    limit_ = path_.speedLimit(cruise_, latAccel_, brakeComfort_);
}

void Driver::link(World& w, const std::string& owner) {
    if (!leaderId_.empty()) leader_ = w.entity(leaderId_, owner + ".driver.leader");
    if (!signalId_.empty()) signal_ = w.entityAs<TrafficSignal>(signalId_, owner + ".driver.signal");
    for (const auto& id : yieldIds_) yield_.push_back(w.entityAs<Walker>(id, owner + ".driver.yield_to"));
}

bool Driver::walkerInBand(const Walker& p, double s0, double s1) const {
    const Vec3 pos = p.pose().p;
    const double s = path_.project(pos);
    return s >= s0 - 0.2 && s <= s1 + 0.2 && std::fabs(path_.lateral(pos)) <= laneWidth_ / 2 + 1.0;
}

double Driver::idmAccel(World& w, double front, double speed, const VehicleLimits& lim) {
    const double v0 = std::max(limit_[size_t(std::clamp(int(s_), 0, int(limit_.size()) - 1))], 0.1);
    double acc = accelMax_ * (1 - std::pow(speed / v0, 4));
    std::vector<std::pair<double, double>> gaps;  // (gap to obstacle, obstacle speed)
    if (signal_ && stopS_ && !passedStop_ && !runsRed_) {
        const std::string& light = signal_->get().state(w.time());
        if (light != "yellow") yellowGo_.reset();
        const double dist = *stopS_ - front;
        if (light == "yellow" && !yellowGo_) yellowGo_ = speed * speed / (2 * std::max(dist, 0.01)) > brakeYellow_;
        if (light == "red" || (light == "yellow" && !*yellowGo_)) gaps.push_back({dist, 0});
    }
    for (const auto& [s0, s1] : crosswalks_) {
        if (front > s0 - 0.3) continue;  // already on or past it
        for (const Walker& p : yield_)
            if (walkerInBand(p, s0, s1)) gaps.push_back({s0 - 0.5 - front, 0});
    }
    if (leader_) {
        const Entity& l = leader_->get();
        const Pose lp = l.pose();
        const double leaderS = path_.project(lp.p);
        const double rear = leaderS - lim.lengthM / 2;  // leader assumed about as long as this vehicle
        if (leaderS > s_) gaps.push_back({rear - front, std::max(0.0, l.velocity().dot(lp.q.axisX()))});
    }
    const double root = std::sqrt(accelMax_ * brakeComfort_);
    for (const auto& [gap, vLead] : gaps) {
        const double want = gapMin_ + speed * headway_ + speed * (speed - vLead) / (2 * root);
        acc -= accelMax_ * std::pow(std::max(want, 0.0) / std::max(gap, 0.05), 2);
    }
    return std::clamp(acc, -brakeMax_, accelMax_);
}

DriveCommand Driver::update(World& w, const std::string& owner, const Pose& pose, double speed,
                            const VehicleLimits& lim, double dt) {
    s_ = path_.project(pose.p);
    const double front = s_ + lim.lengthM / 2;
    if (!initialised_) {
        targetSpeed_ = speed;
        passedStop_ = stopS_ && front > *stopS_;
        stopped_ = speed < 0.05;
        initialised_ = true;
    }
    if (speed < 0.05 && !stopped_) {
        stopped_ = true;
        w.events().emit(owner + ":stops", w.time());
    } else if (speed > 0.3 && stopped_) {
        stopped_ = false;
        w.events().emit(owner + ":moves", w.time());
    }
    if (stopS_ && !passedStop_ && front > *stopS_) {
        passedStop_ = true;
        w.events().emit(owner + ":passes_stop_line", w.time(),
                        {{"light", signal_ ? signal_->get().state(w.time()) : "none"}});
    }
    DriveCommand cmd;
    // Longitudinal: IDM gives the acceleration; a speed loop on the integrated target removes drift.
    const double acc = idmAccel(w, front, speed, lim);
    targetSpeed_ = std::max(0.0, targetSpeed_ + acc * dt);
    const double force = lim.massKg * (acc + 1.5 * (targetSpeed_ - speed));
    if (force >= 0) cmd.throttle = std::clamp(force / lim.maxDriveForceN, 0.0, 1.0);
    else cmd.brake = std::clamp(-force / lim.maxBrakeForceN, 0.0, 1.0);
    if (targetSpeed_ < 0.05 && speed < 0.5) cmd.brake = std::max(cmd.brake, 0.3);  // hold still at a stop
    // Lateral: pure pursuit to a point one look-ahead distance down the path.
    const double look = std::max(4.0, 0.8 * speed);
    const Vec3 target = path_.at(s_ + look);
    const Vec3 local = pose.transformInv(Vec3{target.x, target.y, pose.p.z});
    const double alpha = std::atan2(local.y, std::max(local.x, 0.1));
    cmd.steer = std::clamp(std::atan(2 * lim.wheelbaseM * std::sin(alpha) / look) / lim.maxSteerRad, -1.0, 1.0);
    return cmd;
}

Json Driver::toJson() const {
    Json path = Json::array();
    const double step = std::max(1.0, path_.length() / 64);
    for (double s = 0; s < path_.length(); s += step) path.push_back({path_.at(s).x, path_.at(s).y});
    path.push_back({path_.at(path_.length()).x, path_.at(path_.length()).y});
    Json j = {{"path", path}, {"cruise_mps", cruise_}, {"leader", leaderId_}, {"signal", signalId_}, {"yield_to", yieldIds_}};
    if (stopS_) j["stop_s"] = *stopS_;
    if (runsRed_) j["runs_red"] = true;
    return j;
}

}  // namespace p2s
