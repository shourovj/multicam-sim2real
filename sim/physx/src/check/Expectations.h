// What the prompt asks for, as the scene's "expect" list, checked against X; and camera coverage computed from
// geometry alone (frustum corner rays, projection, occlusion rays): no rendering.
//
//   "expect": [
//     {"id": "crash", "type": "event", "event": "contact:tank:truck", "within_s": [1.0, 2.0]},
//     {"type": "no_event", "event": "contact:car:ped_a"},
//     {"type": "event_order", "events": ["tl:red", "car:stops"]},
//     {"type": "in_view", "camera": "cctv", "entity": "truck", "during_s": [1, 3], "min_fraction": 0.9,
//      "min_screen_fraction": 0.01, "max_occluded": 0.5},
//     {"type": "enters_view", "camera": "cctv", "entity": "car", "outside_at_s": 0.5, "inside_by_s": 3.0},
//     {"type": "leaves_view", "camera": "cctv", "entity": "car", "inside_at_s": 9.0, "outside_by_s": 14.0},
//     {"type": "event_in_view", "camera": "phone", "event": "contact:tank:truck"},
//     {"type": "reaches", "entity": "drone", "point": [8, 3, 6], "within_m": 0.5, "by_s": 7},
//     {"type": "speed", "entity": "car", "during_s": [0, 2], "min_mps": 8, "max_mps": 12},
//     {"type": "at_rest", "entity": "crate", "after_s": 4, "max_speed_mps": 0.05},
//     {"type": "min_gap", "a": "car", "b": "ped_a", "metres": 1.0},
//     {"type": "upright", "entity": "truck", "max_tilt_deg": 30},
//     {"type": "deformation", "entity": "truck_front", "min_m": 0.1, "max_m": 0.5},
//     {"type": "particles_in_region", "entity": "water", "min": [x, y, z], "max": [x, y, z], "at_s": 5, "min_fraction": 0.8},
//     {"type": "obeys_signals"}]
#pragma once

#include "check/Report.h"
#include "check/RunData.h"

namespace p2s {

void runCameraCoverage(const RunData& run, const Spec& scene, Report& report);
void runExpectations(const RunData& run, const Spec& scene, Report& report);

}  // namespace p2s
