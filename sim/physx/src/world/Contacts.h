// Contact reports between entities: the first touch of every pair becomes the event "contact:<a>:<b>" (ids
// sorted), and the impulses are summed per pair so the checker can bound forces in collisions.
#pragma once

#include <functional>
#include <map>
#include <string>
#include <utility>

#include "physx/Px.h"
#include "world/Events.h"

namespace p2s {

struct PairStats {
    double firstT = -1, lastT = -1;
    double totalImpulse = 0;    // N s
    double peakForce = 0;       // N, impulse of one step divided by the step
    int touches = 0;
    double stepImpulse = 0;     // accumulated during the current step
};

class ContactReporter : public physx::PxSimulationEventCallback {
public:
    using NameOf = std::function<std::string(uint32_t entityIndex)>;

    ContactReporter(EventLog& events, NameOf nameOf) : events_(events), nameOf_(std::move(nameOf)) {}
    void beginStep(double t) { t_ = t; }
    void endStep(double dt);  // converts this step's impulses to forces
    // Reports a touch found outside PhysX's rigid contact pipeline (deformables, particles).
    void touch(const std::string& a, const std::string& b, double t, double impulse = 0);
    const std::map<std::pair<std::string, std::string>, PairStats>& pairs() const { return pairs_; }
    Json toJson() const;

    void onContact(const physx::PxContactPairHeader& header, const physx::PxContactPair* pairs, physx::PxU32 n) override;
    void onConstraintBreak(physx::PxConstraintInfo*, physx::PxU32) override {}
    void onWake(physx::PxActor**, physx::PxU32) override {}
    void onSleep(physx::PxActor**, physx::PxU32) override {}
    void onTrigger(physx::PxTriggerPair*, physx::PxU32) override {}
    void onAdvance(const physx::PxRigidBody* const*, const physx::PxTransform*, physx::PxU32) override {}

private:
    PairStats& pair(const std::string& a, const std::string& b);

    EventLog& events_;
    NameOf nameOf_;
    double t_ = 0;
    std::map<std::pair<std::string, std::string>, PairStats> pairs_;
};

// Filter shader: word0 = category bit, word1 = categories ignored, word2 = entity index, word3 = assembly id.
physx::PxFilterFlags filterShader(physx::PxFilterObjectAttributes a0, physx::PxFilterData d0,
                                  physx::PxFilterObjectAttributes a1, physx::PxFilterData d1,
                                  physx::PxPairFlags& flags, const void*, physx::PxU32);

}  // namespace p2s
