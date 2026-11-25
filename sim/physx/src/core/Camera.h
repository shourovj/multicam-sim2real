// Cameras, posed once per frame by the simulator and stored in X, so the checker and Blender use identical
// views. Pose convention is Blender's: the camera looks down its local -Z with +Y up.
//
//   {"id", "type": "fixed",    "position", "look_at" | "look_at_entity"}
//   {"id", "type": "attached", "parent", "offset": {"pos", "yaw_deg", "pitch_deg", "roll_deg"}}   looks along parent +X
//   {"id", "type": "handheld", "position" | "parent"+"offset", "look_at" | "look_at_entity",
//                               "shake": {"rot_deg": 0.6, "pos_m": 0.004, "seed": 11}}
//   {"id", "type": "orbit",    "center" | "center_entity", "radius_m", "height_m", "deg_per_s", "start_deg"}
//   {"id", "type": "follow",   "target", "offset": [x, y, z] (world axes), "lag_s": 0.3}
// Common: "fov_deg" (horizontal, default 60), "resolution": [1280, 720], "clip_m": [0.1, 1000].
#pragma once

#include <functional>
#include <optional>
#include <string>

#include "core/Spec.h"

namespace p2s {

struct Intrinsics {
    int width = 1280, height = 720;
    double fovDeg = 60, nearM = 0.1, farM = 1000;
    double focalPx() const { return width / 2.0 / std::tan(deg2rad(fovDeg) / 2.0); }
};

// Rotation of a camera at `eye` looking at `target` (Blender convention), keeping world +Z up.
Quat lookRotation(const Vec3& eye, const Vec3& target);

class Camera {
public:
    using PoseOf = std::function<Pose(const std::string& entityId)>;

    explicit Camera(const Spec& s);
    const std::string& id() const { return id_; }
    const Intrinsics& intrinsics() const { return intr_; }
    std::vector<std::string> referencedEntities() const;

    // Advances to time t (call once per frame, in order) and returns the camera pose.
    Pose update(double t, const PoseOf& poseOf);
    Json toJson() const;

private:
    Vec3 target(const PoseOf& poseOf) const;
    Pose shake(double t) const;

    std::string id_, type_, parent_, lookEntity_, centerEntity_;
    Json spec_;
    Intrinsics intr_;
    Vec3 position_, lookAt_, center_, offsetPos_;
    Quat offsetRot_;
    double radius_ = 10, height_ = 5, degPerS_ = 10, startDeg_ = 0, lagS_ = 0.3;
    double shakeRot_ = 0, shakePos_ = 0;
    unsigned shakeSeed_ = 11;
    std::optional<Vec3> followPos_;
    double lastT_ = 0;
};

// Projects a world point; returns pixel (u, v) and depth (> 0 in front of the camera).
struct Projection {
    double u, v, depth;
    bool inFrame(const Intrinsics& k) const { return depth > k.nearM && u >= 0 && v >= 0 && u <= k.width && v <= k.height; }
};
Projection project(const Pose& cam, const Intrinsics& k, const Vec3& world);

}  // namespace p2s
