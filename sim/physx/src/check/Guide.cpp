#include "check/Guide.h"

#include <map>
#include <sstream>

namespace p2s {

namespace {

struct Advice {
    const char* meaning;
    const char* fixes;
};

// Keyed by check id (the part of a finding key before ':'); expectation checks share "expect_<type>".
const std::map<std::string, Advice>& advice() {
    static const std::map<std::string, Advice> table = {
        {"simulation",
         {"p2s_sim did not produce a complete X: the scene JSON was rejected, the solver failed, or the run was "
          "interrupted.",
          "Read p2s_sim's SCENE ERROR line: it names the JSON path (e.g. scene.bodies[2].mass_kg). Fix that key; "
          "unknown keys are typos or keys that entity kind does not take. Without one, re-run p2s_sim."}},
        {"finite",
         {"A position, velocity or energy became NaN or infinite: the solver blew up.",
          "Look for huge forces (push, impulse, motor max_force), zero or tiny masses, overlapping bodies at t=0, "
          "or a too-large time step; raise clock.substeps (e.g. 20 -> 40) or add a refine window."}},
        {"physx_errors",
         {"PhysX reported errors (invalid parameters or operations) during the run.",
          "The messages are listed in the finding. Usually a degenerate shape, a joint between a body and itself, or "
          "an action on a body that cannot take it."}},
        {"initial_overlap",
         {"Two bodies overlap in the first frame. PhysX pushes them apart violently, injecting energy.",
          "Move one of them (pos), or shrink its shapes, so they start apart or just touching. Bodies resting on "
          "the ground need their lowest point at the ground height, not below."}},
        {"penetration",
         {"Two solids interpenetrated deeper than the tolerance during the run (tunnelling or crushing through).",
          "Fast impacts need finer steps: add a refine window {between: [a, b], within_m: 1, factor: 8} or raise "
          "clock.substeps; enable ccd on small fast bodies; thicken thin walls (>= 2 cm); check masses are realistic "
          "(a 600 t body against 1 t armour will push through)."}},
        {"below_ground",
         {"Part of a body went below the ground.",
          "Start bodies above the ground; for vehicles check wheels.radius_m and suspension.travel_m; for fast falls "
          "raise substeps or enable ccd."}},
        {"escaped",
         {"A body or particles left the scene (very far away or far below the ground).",
          "Usually an explosion from overlap at t=0, a huge impulse, or particles leaking through thin geometry. "
          "Fix the cause shown alongside; give containers walls at least two particle spacings thick."}},
        {"energy",
         {"The scene gained mechanical energy beyond what its actuators delivered: energy from nothing.",
          "Common causes: overlapping bodies at t=0, joints fighting each other, restitution > 1 combinations, "
          "too-coarse steps in stiff contacts (raise substeps / refine), FEM bodies with very high youngs_pa."}},
        {"acceleration",
         {"A body's peak acceleration is very high. Fine in a crash, suspicious otherwise.",
          "If the peak is not at an impact, look for overlap, a too-stiff joint spring or an impulse in the wrong "
          "units (impulse is N s, push is N). Acknowledge it if it is a genuine collision."}},
        {"speed",
         {"A body moved faster than is plausible for its kind.",
          "Check initial velocities, controller targets (cruise_mps, speed_mps) and units (m/s, not km/h)."}},
        {"actuator",
         {"An actuator's peak force or power is outside the usual range for its kind (table 'Actuators').",
          "Adjust limits (max_wheel_torque_nm, rotor.thrust_to_weight, motor.max_force, traction.mu) or masses so the "
          "numbers match the real machine; acknowledge with a reason if the machine really is that strong."}},
        {"jitter",
         {"A body that should be resting keeps vibrating.",
          "Raise solver_iterations, lower max_depenetration_mps, make sure it does not start overlapping its support."}},
        {"fem_inverted",
         {"A soft body's tetrahedra turned inside out: the FEM solve failed.",
          "Lower the impact speed or raise youngs_pa / solver_iterations / resolution; use plastic for permanent dents "
          "instead of very soft elastic material."}},
        {"particles_in_solids",
         {"Many particles ended up inside solid bodies (leaking or tunnelling).",
          "Thicken walls to >= 2 particle spacings, lower emitter speed_mps, raise substeps."}},
        {"expect_coverage",
         {"The scene does not state enough expectations for the checker to confirm it matches the prompt.",
          "Add to expect: at least one in_view or enters_view per camera, and at least one physical outcome (event, "
          "reaches, at_rest, deformation, particles_in_region, ...) per thing the prompt says happens."}},
        {"expect_event",
         {"An expected event did not happen, or not within its time window.",
          "Check the event name in the events list of x.json (contact names sort ids: contact:a:b with a < b). Move "
          "the bodies or change speeds so it happens in the window; or widen within_s if the prompt allows."}},
        {"expect_no_event",
         {"An event that must not happen did happen (e.g. a car touched a pedestrian).",
          "Change paths, start times, speeds or add yield_to / a signal so the event is avoided."}},
        {"expect_event_order",
         {"Events happened in a different order than expected.",
          "Adjust timings: trigger delay_s, start_s, initial positions or speeds."}},
        {"expect_in_view",
         {"An entity is not (enough) in a camera's view when it should be: out of frame, too small, or occluded.",
          "Use the camera coverage table: aim the camera (look_at / look_at_entity), move it closer, widen fov_deg, or "
          "move occluders. Corner-ray ground footprints show what each camera sees."}},
        {"expect_enters_view",
         {"An entity was supposed to come into a camera's view during the clip but did not (or was visible too early).",
          "Shift the entity's start position or the camera so it starts outside the frustum and crosses into it before "
          "inside_by_s."}},
        {"expect_leaves_view",
         {"An entity was supposed to leave a camera's view but stayed in it.",
          "Extend its path or speed, or narrow/turn the camera."}},
        {"expect_event_in_view",
         {"An event happened off-screen for a camera that should show it.",
          "Aim the camera at the event location (listed in the finding) at its time; a follow or orbit camera may help."}},
        {"expect_reaches",
         {"An entity did not get within the required distance of a point by the deadline.",
          "Check its goals / path / speed; for drones the goal t; for vehicles cruise_mps and path length."}},
        {"expect_speed",
         {"An entity's speed left the expected range during the window.", "Tune its controller targets or initial velocity."}},
        {"expect_at_rest",
         {"An entity was still moving after it should have come to rest.",
          "Raise friction or damping, lower speeds, or extend the clip."}},
        {"expect_min_gap",
         {"Two entities came closer than allowed.", "Separate their paths or timings; for traffic use leader / signals / yield_to."}},
        {"expect_upright",
         {"An entity tilted more than allowed (rolled over, tipped).",
          "Lower its centre of mass (com), widen the track, lower speed in bends (lat_accel_max)."}},
        {"expect_deformation",
         {"A soft body's permanent deformation is outside the expected range.",
          "Lower youngs_pa or plastic.yield_strain for more crush, raise them for less; impact speed and mass matter."}},
        {"expect_particles_in_region",
         {"Too few particles ended up in the expected region (e.g. water in the glass).",
          "Aim the emitter (pos, direction, speed_mps), start it at the right time, or move the container."}},
        {"expect_obeys_signals",
         {"A vehicle crossed its stop line on red.", "Give its driver the signal and stop_s; check the plan timings."}},
    };
    return table;
}

}  // namespace

std::string nextStep(Status status) {
    switch (status) {
        case Status::Pass:
            return "**Next:** stop iterating. The physics output X is accepted; hand it to rendering (sim/blender).";
        case Status::Fail:
            return "**Next:** edit the scene JSON to fix every failure and every unacknowledged warning below, bump "
                   "meta.iteration, re-run p2s_sim and p2s_check. Change only what a finding points at; keep what passed.";
        case Status::StopMaxIterations:
        case Status::StopStuck:
            return "**Next:** stop iterating and report to a human: the remaining findings, what you tried (see "
                   "Previous iterations), and your best guess at the cause. Do not render this scene.";
    }
    return "";
}

std::string iterationGuide(const std::set<std::string>& failingChecks) {
    std::ostringstream md;
    md << "## Iteration guide\n\n"
          "How to read this report: findings are FAIL (must fix), WARN (fix, or acknowledge in the scene's "
          "`acknowledge` list as {\"check\": \"<key>\", \"reason\": \"<why it is right, >= 10 chars>\"}), INFO and PASS. "
          "The loop ends when STATUS is PASS (no failures, every warning fixed or acknowledged, every expectation met) "
          "or a STOP status. Tables are measurements for you to judge against the prompt and the real world; numbers "
          "are SI. Fix the earliest physical cause first: an overlap at t=0 causes energy, acceleration and "
          "penetration findings downstream.\n\n";
    if (failingChecks.empty()) return md.str();
    md << "What the failing checks measure and how to fix them:\n\n";
    for (const std::string& c : failingChecks)
        if (const auto it = advice().find(c); it != advice().end())
            md << "- **" << c << "**: " << it->second.meaning << " " << it->second.fixes << "\n";
    return md.str() + "\n";
}

}  // namespace p2s
