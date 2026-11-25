// A body of water with a flat surface: world.water = {"level": 0, "density": 1000, "current": [vx, vy, vz],
// "min": [x, y], "max": [x, y]} (optional horizontal extent; unbounded without it).
#pragma once

#include <optional>

#include "core/Spec.h"

namespace p2s {

struct Water {
    double level = 0, density = 1000;
    Vec3 current;
    std::optional<Aabb> extent;  // horizontal bounds; z ignored

    static std::optional<Water> parse(const Spec& s) {
        Water w;
        w.level = s.num("level", 0);
        w.density = s.positive("density", 1000);
        w.current = s.vec3("current", {0, 0, 0});
        if (s.has("min") || s.has("max")) {
            const auto lo = s.numbers("min"), hi = s.numbers("max");
            if (lo.size() != 2 || hi.size() != 2 || hi[0] <= lo[0] || hi[1] <= lo[1]) s.fail("min and max are [x, y] with max > min");
            Aabb box;
            box.add({lo[0], lo[1], -1e9});
            box.add({hi[0], hi[1], 1e9});
            w.extent = box;
        }
        return w;
    }
    bool wet(const Vec3& p) const { return p.z < level && (!extent || extent->contains(p)); }
    Json toJson() const {
        Json j = {{"level", level}, {"density", density}, {"current", p2s::toJson(current)}};
        if (extent) j["min"] = {extent->lo.x, extent->lo.y}, j["max"] = {extent->hi.x, extent->hi.y};
        return j;
    }
};

}  // namespace p2s
