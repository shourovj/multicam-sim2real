// What a camera sees of an entity at a frame, from geometry alone: the entity's surface sample points projected into
// the image, within the clip range, with occlusion rays against every other rigid part (the camera's own parent never
// occludes). Shared by the camera coverage tables, the scene's expectations and the prompt-spec checks.
#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "check/RunData.h"

namespace p2s {

struct View {
    double inFrame = 0;   // fraction of the entity's sample points inside the image
    double visible = 0;   // fraction inside the image and not hidden behind another body
    double area = 0;      // screen-space bounding box of the visible points, as a fraction of the image
    double widthPx = 0;   // that box in pixels
    double heightPx = 0;
};

inline bool inClip(const Projection& pr, const Intrinsics& k) { return pr.inFrame(k) && pr.depth <= k.farM; }

// Visibility of every entity from every camera, computed on demand and cached.
class Visibility {
public:
    explicit Visibility(const RunData& run) : run_(run), bounds_(size_t(run.frames())) {}
    const View& at(const CameraTrack& cam, const std::string& entity, int f);

private:
    const std::vector<Aabb>& bounds(int f);
    View compute(const CameraTrack& cam, const std::string& entity, int f);

    const RunData& run_;
    std::vector<std::vector<Aabb>> bounds_;  // per frame: every part's world bounds, filled on first use
    std::map<std::pair<std::string, std::string>, std::vector<std::optional<View>>> cache_;
};

}  // namespace p2s
