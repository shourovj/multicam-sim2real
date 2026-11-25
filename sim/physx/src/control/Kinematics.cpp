#include "control/Kinematics.h"

#include <fstream>
#include <map>

#include "control/Linear.h"

namespace p2s {

namespace {

Vec3 v3(const Json& j) { return {j[0].get<double>(), j[1].get<double>(), j[2].get<double>()}; }

Pose jointMotion(const JointModel& j, double q) {
    return j.revolute ? Pose{{}, Quat::axisAngle(j.axis, q)} : Pose{j.axis * q, {}};
}

}  // namespace

Vec3 rotationVector(const Quat& from, const Quat& to) {
    Quat d = to * from.conj();
    if (d.w < 0) d = {-d.x, -d.y, -d.z, -d.w};
    const double s = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
    if (s < 1e-12) return {};
    const double angle = 2 * std::atan2(s, d.w);
    return Vec3{d.x, d.y, d.z} * (angle / s);
}

Chain Chain::load(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw SpecError("no robot description " + path + " (make one with sim/physx/tools/mjcf_to_robot.py)");
    const Json r = Json::parse(in);
    Chain c;
    std::map<std::string, int> index;
    for (const Json& l : r.at("links")) {
        LinkModel m;
        m.name = l.at("name");
        m.parent = l.at("parent").is_null() ? -1 : index.at(l.at("parent").get<std::string>());
        const Json& q = l.at("quat_xyzw");
        m.local = {v3(l.at("pos")), Quat{q[0], q[1], q[2], q[3]}.normalized()};
        if (l.contains("joint")) {
            const Json& j = l.at("joint");
            m.joint = JointModel{j.at("name"), j.at("type") == "revolute", v3(j.at("axis")).normalized(), j.at("range")[0], j.at("range")[1]};
        }
        m.massKg = l.value("mass_kg", 0.1);
        if (l.contains("com")) m.com = v3(l.at("com"));
        for (const Json& s : l.at("shapes")) {
            Json full = s;
            full["material"] = "plastic";
            m.shapes.push_back(Shape::fromJson(full));
        }
        m.visuals = l.value("visuals", Json::array());
        index[m.name] = int(c.links_.size());
        if (m.joint && m.joint->revolute) c.arm_.push_back(int(c.links_.size()));
        c.links_.push_back(std::move(m));
    }
    c.tcpLink_ = index.at(r.at("tcp").at("link").get<std::string>());
    c.tcpOffset_ = {v3(r.at("tcp").at("offset")), {}};
    for (const Json& h : r.at("home")) c.home_.push_back(h.get<double>());
    if (c.home_.size() != c.arm_.size()) throw SpecError(path + ": home needs one value per revolute joint");
    return c;
}

std::vector<Pose> Chain::forward(const Pose& base, const std::vector<double>& q) const {
    std::vector<Pose> out(links_.size());
    for (size_t i = 0; i < links_.size(); ++i) {
        const LinkModel& l = links_[i];
        Pose p = (l.parent < 0 ? base : out[size_t(l.parent)]) * l.local;
        if (l.joint) p = p * jointMotion(*l.joint, q[i]);
        out[i] = p;
    }
    return out;
}

Pose Chain::tcp(const Pose& base, const std::vector<double>& q) const {
    return forward(base, q)[size_t(tcpLink_)] * tcpOffset_;
}

std::vector<double> Chain::fullQ(const std::vector<double>& arm, std::vector<double> q) const {
    for (size_t k = 0; k < arm_.size(); ++k) q[size_t(arm_[k])] = arm[k];
    return q;
}

Chain::Solution Chain::solve(const Pose& base, const Pose& target, std::vector<double> arm, std::vector<double> fixedQ,
                             int iterations) const {
    const int n = int(arm_.size());
    const double lambda = 0.05, homePull = 0.05;
    for (int it = 0; it < iterations; ++it) {
        const auto q = fullQ(arm, fixedQ);
        const auto frames = forward(base, q);
        const Pose tcpPose = frames[size_t(tcpLink_)] * tcpOffset_;
        const Vec3 ep = target.p - tcpPose.p, er = rotationVector(tcpPose.q, target.q);
        if (ep.norm() < 2e-4 && er.norm() < 2e-3) break;
        Matrix J(6, n);
        for (int k = 0; k < n; ++k) {
            const Pose& jf = frames[size_t(arm_[size_t(k)])];
            const Vec3 a = jf.q.rotate(links_[size_t(arm_[size_t(k)])].joint->axis);
            const Vec3 lin = a.cross(tcpPose.p - jf.p);
            J(0, k) = lin.x, J(1, k) = lin.y, J(2, k) = lin.z, J(3, k) = a.x, J(4, k) = a.y, J(5, k) = a.z;
        }
        const Matrix pinv = J.pseudoInverse(lambda);
        std::vector<double> dq = pinv * std::vector<double>{ep.x, ep.y, ep.z, er.x, er.y, er.z};
        // Pull toward the home posture in the null space, so the elbow stays natural.
        const Matrix exact = J.pseudoInverse(1e-6) * J;
        std::vector<double> pull(static_cast<size_t>(n));
        for (int k = 0; k < n; ++k) pull[size_t(k)] = homePull * (home_[size_t(k)] - arm[size_t(k)]);
        const std::vector<double> projected = exact * pull;
        for (int k = 0; k < n; ++k) {
            const JointModel& jm = *links_[size_t(arm_[size_t(k)])].joint;
            const double step = std::clamp(dq[size_t(k)] + pull[size_t(k)] - projected[size_t(k)], -0.2, 0.2);
            arm[size_t(k)] = std::clamp(arm[size_t(k)] + step, jm.lower + 1e-3, jm.upper - 1e-3);
        }
    }
    const Pose tcpPose = tcp(base, fullQ(arm, fixedQ));
    return {arm, (target.p - tcpPose.p).norm(), rad2deg(rotationVector(tcpPose.q, target.q).norm())};
}

}  // namespace p2s
