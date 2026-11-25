// A serial robot as a tree of links (from sim/physx/robots/<name>.json): forward kinematics, the geometric
// Jacobian of the tool centre point (TCP), and damped least-squares inverse kinematics with joint limits and a
// pull toward the home posture inside the null space (ported from sim/physics/arm.py).
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "core/Shape.h"

namespace p2s {

struct JointModel {
    std::string name;
    bool revolute = true;  // else prismatic
    Vec3 axis;             // in the link frame
    double lower = 0, upper = 0;
};

struct LinkModel {
    std::string name;
    int parent = -1;
    Pose local;  // in the parent link frame, at joint value 0
    std::optional<JointModel> joint;
    double massKg = 0;
    Vec3 com;
    std::vector<Shape> shapes;
    Json visuals;
};

class Chain {
public:
    static Chain load(const std::string& path);  // throws SpecError

    const std::vector<LinkModel>& links() const { return links_; }
    const std::vector<int>& armJoints() const { return arm_; }  // link indices of the revolute joints, root first
    int tcpLink() const { return tcpLink_; }
    const Pose& tcpOffset() const { return tcpOffset_; }
    const std::vector<double>& home() const { return home_; }  // one value per arm joint

    // World poses of every link for joint values q (one per link; links without a joint ignore theirs).
    std::vector<Pose> forward(const Pose& base, const std::vector<double>& q) const;
    Pose tcp(const Pose& base, const std::vector<double>& q) const;
    // Arm joint values (warm-started from `arm`) that put the TCP at `target`; returns the residual error.
    struct Solution {
        std::vector<double> arm;
        double errorM, errorDeg;
    };
    Solution solve(const Pose& base, const Pose& target, std::vector<double> arm, std::vector<double> fixedQ, int iterations) const;
    std::vector<double> fullQ(const std::vector<double>& arm, std::vector<double> q) const;

private:
    std::vector<LinkModel> links_;
    std::vector<int> arm_;
    int tcpLink_ = -1;
    Pose tcpOffset_;
    std::vector<double> home_;
};

// Rotation vector (axis * angle) taking orientation `from` to `to`.
Vec3 rotationVector(const Quat& from, const Quat& to);

}  // namespace p2s
