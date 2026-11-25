// A finished simulation (X) loaded for checking: every rigid part's shapes and per-frame poses, deformable and
// particle point sets, camera tracks, events, contacts, actuators and energy. No PhysX.
#pragma once

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "core/Camera.h"
#include "core/Recorder.h"
#include "core/Shape.h"

namespace p2s {

// One rigid piece that moves as a unit: a body, a vehicle chassis, a wheel.
struct Part {
    std::string entity, name, kind;  // kind: the entity's kind, or "wheel"
    std::vector<Shape> shapes;
    std::vector<Pose> poses;         // per frame
    std::vector<Vec3> samples;       // surface sample points in the part frame
    Aabb local;                      // bounds in the part frame
    double minHalfExtent = 1e9;      // of the thinnest shape: penetration tolerance scales with it

    Aabb worldBounds(int frame) const { return local.transformed(poses[size_t(frame)]); }
    double sdf(int frame, const Vec3& world) const { return PosedShapes{shapes, poses[size_t(frame)]}.sdf(world); }
};

// Deformable or particle matter: a cloud of points per frame.
struct PointSet {
    std::string entity, kind, host;  // kind: soft_body, cloth, particles
    std::vector<std::vector<Vec3>> frames;
    std::vector<Vec3> rest;          // soft bodies: rest positions (host frame if hosted)
    std::vector<std::array<int32_t, 4>> tets;
    double radius = 0;               // particles: particle radius
    std::string particleKind;        // fluid, granular, smoke
};

struct CameraTrack {
    std::string id;
    std::string parent;  // the body it rides on (dashcam's car): never an occluder, as it is not rendered in view
    Intrinsics k;
    std::vector<Pose> poses;
};

struct EntityInfo {
    std::string id, kind, assembly;
    Json desc;
    std::vector<Vec3> velocity;      // per frame, if recorded
    double mass = 0;
};

class RunData {
public:
    explicit RunData(const std::string& dir);

    const Json& x() const { return x_; }
    int frames() const { return frames_; }
    double fps() const { return fps_; }
    double time(int f) const { return f / fps_; }
    int frameAt(double t) const;
    double groundZ() const { return groundZ_; }
    bool hasGround() const { return hasGround_; }

    const std::vector<Part>& parts() const { return parts_; }
    const std::vector<PointSet>& pointSets() const { return points_; }
    const std::vector<CameraTrack>& cameras() const { return cameras_; }
    const std::map<std::string, EntityInfo>& entities() const { return entities_; }
    bool hasEntity(const std::string& id) const { return entities_.count(id) > 0; }
    const EntityInfo& entity(const std::string& id) const;  // throws SpecError if unknown
    // Entities that may legitimately overlap: same entity, same assembly, joined, or host and mounted deformable.
    bool exempt(const std::string& a, const std::string& b) const;
    // World position of an entity at a frame: its first part's pose, or a point set's centroid.
    std::optional<Vec3> position(const std::string& id, int frame) const;
    // Surface sample points of an entity in world space at a frame (at most `maxPoints`).
    std::vector<Vec3> samples(const std::string& id, int frame, size_t maxPoints = 400) const;
    // First time of an event; nullopt if it never happened.
    std::optional<double> eventTime(const std::string& name) const;
    std::vector<std::array<double, 4>> energy() const;  // per frame: kinetic, potential, net work, work in

private:
    void loadParts(const Json& e, const EntityInfo& info, const StreamReader& sr);
    void loadPoints(const Json& e, const StreamReader& sr);

    Json x_;
    std::string dir_;
    int frames_ = 0;
    double fps_ = 24, groundZ_ = 0;
    bool hasGround_ = false;
    std::vector<Part> parts_;
    std::vector<PointSet> points_;
    std::vector<CameraTrack> cameras_;
    std::map<std::string, EntityInfo> entities_;
    std::set<std::pair<std::string, std::string>> exempt_;
};

}  // namespace p2s
