// A traffic signal with a fixed-time plan (from sim/world/signals.py). Not a body: drivers read its state.
//   {"id": "tl_NS", "plan": [{"state": "green", "until_s": 3.0}, {"state": "yellow", "until_s": 4.0},
//                            {"state": "red", "until_s": 7.2}],
//    "heads": [{"pos": [x, y, z], "yaw_deg": 0}]}      where the lamps are, for the renderer
// Events: <id>:<state> at every change. Stream signal/<id>: 0 red, 1 yellow, 2 green.
#pragma once

#include "world/World.h"

namespace p2s {

class TrafficSignal : public Entity {
public:
    TrafficSignal(const Spec& s, World& w);

    const std::string& state(double t) const;
    void afterStep(World& w, double dt) override;
    void record(World& w, Recorder& rec) override;
    Json describe() const override;

private:
    struct Phase {
        std::string state;
        double until;
    };
    std::vector<Phase> plan_;
    std::vector<Pose> heads_;
    std::string last_;
};

}  // namespace p2s
