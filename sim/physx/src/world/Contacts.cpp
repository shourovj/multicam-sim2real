#include "world/Contacts.h"

#include <array>

using namespace physx;

namespace p2s {

PxFilterFlags filterShader(PxFilterObjectAttributes, PxFilterData d0, PxFilterObjectAttributes, PxFilterData d1,
                           PxPairFlags& flags, const void*, PxU32) {
    if (d0.word3 != 0 && d0.word3 == d1.word3) return PxFilterFlag::eKILL;  // same assembly
    if ((d0.word0 & d1.word1) || (d1.word0 & d0.word1)) return PxFilterFlag::eKILL;
    flags = PxPairFlag::eCONTACT_DEFAULT | PxPairFlag::eNOTIFY_TOUCH_FOUND | PxPairFlag::eNOTIFY_TOUCH_PERSISTS |
            PxPairFlag::eNOTIFY_CONTACT_POINTS;
    return PxFilterFlag::eDEFAULT;
}

PairStats& ContactReporter::pair(const std::string& a, const std::string& b) {
    return pairs_[a < b ? std::make_pair(a, b) : std::make_pair(b, a)];
}

void ContactReporter::touch(const std::string& a, const std::string& b, double t, double impulse) {
    if (a == b) return;
    PairStats& p = pair(a, b);
    if (p.firstT < 0) {
        p.firstT = t;
        const auto& [x, y] = a < b ? std::make_pair(a, b) : std::make_pair(b, a);
        events_.emit("contact:" + x + ":" + y, t);
    }
    p.lastT = t;
    ++p.touches;
    p.stepImpulse += impulse;
}

void ContactReporter::endStep(double dt) {
    for (auto& [_, p] : pairs_) {
        p.totalImpulse += p.stepImpulse;
        p.peakForce = std::max(p.peakForce, p.stepImpulse / dt);
        p.stepImpulse = 0;
    }
}

void ContactReporter::onContact(const PxContactPairHeader& header, const PxContactPair* pairs, PxU32 n) {
    if (header.flags & (PxContactPairHeaderFlag::eREMOVED_ACTOR_0 | PxContactPairHeaderFlag::eREMOVED_ACTOR_1)) return;
    std::array<PxContactPairPoint, 64> pts;
    for (PxU32 i = 0; i < n; ++i) {
        const PxContactPair& cp = pairs[i];
        const std::string a = nameOf_(cp.shapes[0]->getSimulationFilterData().word2);
        const std::string b = nameOf_(cp.shapes[1]->getSimulationFilterData().word2);
        double impulse = 0;
        const PxU32 k = cp.extractContacts(pts.data(), PxU32(pts.size()));
        for (PxU32 j = 0; j < k; ++j) impulse += pts[j].impulse.magnitude();
        touch(a, b, t_, impulse);
    }
}

Json ContactReporter::toJson() const {
    Json out = Json::array();
    for (const auto& [k, p] : pairs_)
        out.push_back({{"a", k.first},
                       {"b", k.second},
                       {"first_t", p.firstT},
                       {"last_t", p.lastT},
                       {"total_impulse_ns", p.totalImpulse},
                       {"peak_force_n", p.peakForce},
                       {"touch_steps", p.touches}});
    return out;
}

}  // namespace p2s
