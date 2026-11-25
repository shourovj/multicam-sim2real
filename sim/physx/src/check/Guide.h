// The iteration guide appended to every report: what to do next, and for each failing check what it measures
// and which scene keys usually fix it. Generated, so it always matches the checks that exist.
#pragma once

#include <set>
#include <string>

#include "check/Report.h"

namespace p2s {

std::string nextStep(Status status);
std::string iterationGuide(const std::set<std::string>& failingChecks);

}  // namespace p2s
