# sim/physx: prompt to sim (p2s)

GPU physics for the pipeline's first stage. A SceneAgent (an LLM) turns a prompt into a scene file S (JSON);
`p2s_sim` simulates it on NVIDIA PhysX 5.9 and writes the physics output X; `p2s_check` checks S and X and writes the
report r with a `STATUS` that ends or continues the agent's loop. Nothing is rendered while iterating: camera
coverage comes from geometry (corner rays, projection, occlusion rays). After a PASS, `sim/blender/import_x.py`
builds the Blender scene from X.

```
prompt ─> SceneAgent ─> S ─> p2s_sim ─> X ─> p2s_check ─> r (PASS: render; FAIL: fix S; STOP: ask a human)
```

## Documents

| file | for |
|---|---|
| [docs/...-sceneagent-prompt.md](docs/2026-11-24-1859-sceneagent-prompt.md) | the SceneAgent's prompt: the loop, how to run both programs, the scene format, expectations |
| [docs/...-sceneagent-iteration.md](docs/2026-11-24-1859-sceneagent-iteration.md) | the standing instruction for each iteration (the report adds a generated guide for the checks that failed) |
| [docs/...-x-format.md](docs/2026-11-24-1859-x-format.md) | the physics output X: x.json and the streams |

## Build

```bash
source sim/physx/env.sh
sim/physx/setup_physx.sh                 # once: PhysX 5.9.1 + plasticity patch, CUDA 13 toolkit (into .deps/)
cmake -S sim/physx -B sim/physx/build && cmake --build sim/physx/build -j
CUDA_VISIBLE_DEVICES=<gpu> sim/physx/build/p2s_tests      # 20 tests against analytic physics, ~2 min
```

`.deps/` (gitignored) holds the PhysX source and build, the CUDA 13 runtime, the Python env and the MuJoCo Menagerie
robots. PhysX is patched (`third_party/physx-5.9.1-s2r.patch`): CUDA 13 build fixes and the FEM plasticity kernels,
which stock 5.9.1 compiles but disables.

## Run

```bash
CUDA_VISIBLE_DEVICES=<gpu> sim/physx/build/p2s_sim  scene.json out/     # exit 0 ok, 2 bad scene, 3 solver failure
sim/physx/build/p2s_check scene.json out/                                # exit 0 PASS, 1 FAIL, 3 STOP; writes out/r.md
sim/blender/.deps/blender-4.2.1-linux-x64/blender -b --python sim/blender/import_x.py -- out/ scene.blend
```

## Layout

| path | what |
|---|---|
| `src/core/` | no PhysX: math, the JSON reader (rejects unknown keys, errors name the JSON path), shapes with exact signed distances and ray casts (the one 3D model the simulator collides, the checker measures and Blender draws), cameras, streams, wind |
| `src/physx/` | PhysX glue: owning handles, runtime, materials, shapes |
| `src/world/` | the World (scene, clock, triggers, contact events, refinement windows, recording X) and the Entity base class |
| `src/bodies/` | one class per kind: RigidBody, Joint, Multirotor, Vehicle, TrackedVehicle, RobotArm, Walker, TrafficSignal, SoftBody, Cloth, ParticleSystem; `Registry.cpp` maps scene keys to them |
| `src/control/` | controllers: Driver (lane path, IDM, signals, crosswalks), GoalPath (quintic), Kinematics (robot FK/IK), Linear |
| `src/check/` | the checker: RunData (reads X), PhysicsChecks, Expectations and camera coverage, Report, Guide |
| `app/` | `sim_main.cpp` (p2s_sim), `check_main.cpp` (p2s_check) |
| `tests/` | doctest unit tests: geometry and spec (no GPU), physics against analytic results (GPU) |
| `scenes/` | example scenes that pass the checker, ported from the earlier pipelines |
| `robots/` | robot descriptions made from MJCF by `tools/mjcf_to_robot.py` |
| `tools/` | `mjcf_to_robot.py`, `make_round1_scenes.py` |

## What it simulates

| building block | scene key | ported from |
|---|---|---|
| rigid bodies (dynamic, kinematic paths, static), drag, gyroscopic spin, impulses and pushes | `bodies` | rigid_body.py, tank-truck |
| joints: fixed, hinge, slider, ball, rope/spring, motors, breakable | `joints` | payload carriers |
| multirotors flown by per-rotor thrust | `multirotors` | quadrotor.py |
| wheeled vehicles with suspension, tyres, drivers (IDM, signals, crosswalks, leaders) | `vehicles` | ground_vehicle.py, lane_vehicle.py, tank-truck truck |
| tracked vehicles with traction control and optional route | `tracked_vehicles` | tank-truck tank, convoy |
| robot arms (articulation + IK) carrying objects | `robot_arms` | arm.py |
| pedestrians | `walkers` | pedestrian.py |
| traffic signals | `signals` | signals.py |
| elastic and plastic FEM solids | `soft_bodies` | tank-truck crumple zone and armour |
| cloth in wind | `cloths` | tank-truck flag |
| liquids, sand, smoke (PBD particles, emitters) | `particles` | tank-truck oil, pour.py |
| wind field (gusts, turbulence) | `world.wind` | tank-truck wind |
| triggers on time or events | `triggers` | tank-truck driver reactions |
| crash refinement windows | `refine` | tank-truck crash window |
| cameras: fixed, attached (to parts), handheld, orbit, follow | `cameras` | cameras.py, tank-truck cameras |
