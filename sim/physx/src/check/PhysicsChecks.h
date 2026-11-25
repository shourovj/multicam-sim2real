// Checks every scene must pass regardless of the prompt, generalised from the tank-truck audits:
// finite state, solver errors, interpenetration (at t=0 and during the run), energy not created from nothing,
// escaped bodies, inverted FEM elements, particles inside solids; plus plausibility tables (acceleration,
// speed, actuators) that the LLM judges and the fact card.
#pragma once

#include "check/Report.h"
#include "check/RunData.h"

namespace p2s {

void runPhysicsChecks(const RunData& run, Report& report);

}  // namespace p2s
