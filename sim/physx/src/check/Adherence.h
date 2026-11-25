// Prompt adherence against an independent prompt spec (spec/README.md): Eval1's half that asks "is this the scene the
// prompt describes?", where the physics checks ask "is this physically possible?".
//
// The scene's own `expect` list is written by the SceneAgent, so it can only confirm what the agent remembered to
// state: a crumple zone the agent dropped from S also drops out of its expectations. The prompt spec is written from
// the prompt alone (p2s_spec, once per prompt, never by the SceneAgent) in the spirit of NeuS-V (Sharan et al., CVPR
// 2025): the prompt's objects with their class, count and colour, its parts, its cameras, and one temporal-logic
// formula per requirement over a fixed vocabulary of predicates. Here those predicates are grounded in X exactly
// (poses, contacts, deformation, particles, signals, camera visibility from geometry), so no vision model judges
// anything: each requirement gets a robustness in [-1, 1] (> 0 holds) and a reason when it fails.
//
//   1. Bind: every prompt object to scene entities (kind, size against the class, appearance, id words), every part
//      to the region of its host (and the soft bodies mounted there), every camera to a scene camera.
//   2. Check the objects themselves: present, as many as the prompt says, the right size for the class, the colour
//      and class the prompt gives (appearance), no contact with a person the prompt does not ask for.
//   3. Check each requirement's formula (formula semantics: Formula.h).
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "check/Report.h"
#include "check/RunData.h"

namespace p2s {

// The prompt spec itself is malformed: not a scene problem, so the SceneAgent cannot fix it.
class PromptSpecError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Problems with a spec on its own (no run): format, object names, formulas, predicate names and arguments. Empty if
// it is well formed. p2s_spec uses this (p2s_check --lint-spec) to have the spec writer repair its output.
std::vector<std::string> lintPromptSpec(const Json& spec);

// Binds and checks the spec against S and X, adding findings ("prompt_object", "prompt_appearance", "prompt_part",
// "prompt_camera", "prompt_contact", "prompt") and tables. Throws PromptSpecError for a malformed spec, or when the
// spec was written for another prompt than meta.prompt.
void runAdherence(const RunData& run, const Spec& scene, const Json& spec, Report& report);

}  // namespace p2s
