# SceneAgent: from a prompt to verified physics

You turn a text prompt (a scene and the cameras that film it) into a **scene file S** (JSON). A GPU physics
simulator (PhysX 5.9) runs S and writes the **physics output X**. A checker reads S and X and writes the
**report r**. You iterate on S until r says `STATUS: PASS`. Nothing is rendered while you iterate: everything you
need to know about the result, including what each camera sees, is in r as text and numbers.

```
prompt ──> you write S (scene.json) ──> p2s_sim ──> X (x.json + streams/) ──> p2s_check ──> r (r.md, r.json)
              ^                                                                                 │
              └───────────────────────── fix what r says, iteration += 1 ─────────────────────┘
```

## 1. How to run

```bash
source sim/physx/env.sh                      # paths; once per shell
P=sim/physx/build                            # built by: cmake -S sim/physx -B sim/physx/build && cmake --build sim/physx/build -j
RUN=$S2R_DATA/physx/<scene>                  # one folder per scene
mkdir -p $RUN/it1 && cp scene.json $RUN/it1/scene.json
CUDA_VISIBLE_DEVICES=<free gpu> $P/p2s_sim $RUN/it1/scene.json $RUN/it1      # seconds to minutes
$P/p2s_check $RUN/it1/scene.json $RUN/it1                                     # no GPU; writes $RUN/it1/r.md
```

- `p2s_sim` exits 0 when it simulated, **2** when S is invalid (stderr names the JSON path, e.g.
  `scene.vehicles[0].wheels.radius_m: must be > 0`), 3 on a solver failure.
- `p2s_check` exits 0 **PASS**, 1 **FAIL** (iterate), 3 **STOP** (give up and report to a human), 2 bad input
  (then r.md says `STATUS: ERROR` with the JSON path of the problem, e.g. an expectation naming an unknown entity).
- Keep every iteration in its own folder (`it1`, `it2`, ...) with its own copy of S. All iterations of a scene
  share `$RUN/history.jsonl`, so each report shows how earlier iterations went.
- Pick a free GPU with `nvidia-smi` (use one with < 1 GB used).

## 2. The loop and when it ends

1. Read the prompt. List every object, every thing that happens, and every camera.
2. Write S (sections 4-6). Fill `expect` (section 7) with one entry per thing the prompt says happens or shows.
3. Run both programs. Read `r.md` top to bottom. It has the status, the failures with fixes, measurement tables
   (motion, actuators, what each camera sees), a fact card and an iteration guide for the failing checks.
4. If `FAIL`: change only what the findings point at, set `meta.iteration` one higher, run again in a new folder.
5. Stop when the status is:
   - `PASS`: no failures, every warning fixed or acknowledged, every expectation met. X is ready for rendering.
   - `STOP_MAX_ITERATIONS` (default 8, `meta.max_iterations`) or `STOP_STUCK` (the same findings three times in a
     row): stop, and report the remaining findings and what you tried to a human.

A warning that is right for this scene (a genuine 80 g crash deceleration) can be acknowledged instead of fixed:
`"acknowledge": [{"check": "acceleration:truck", "reason": "head-on crash at 16 m/s; 80 g is expected"}]`. The
`check` value is the bold key of the finding in r.md. Failures cannot be acknowledged.

## 3. Conventions

- World frame: **Z up**, metres, kilograms, seconds, newtons. Angles in **degrees** where the key ends in `_deg`,
  otherwise radians (`angular_velocity`).
- Body frames: +X forward, +Y left, +Z up, origin on the ground for vehicles and people.
- `pos` is a position `[x, y, z]`; orientation is `yaw_deg` (about Z), `pitch_deg` (about Y), `roll_deg`
  (about X), applied yaw then pitch then roll; or `quat_xyzw`.
- Every entity has a unique `id` (no `:`). The ground is the entity `ground`, top surface at z = 0 by default.
- Unknown keys are errors (they are usually typos). Optional keys have the defaults shown.
- Contact events are named `contact:<a>:<b>` with the two ids in alphabetical order.

## 4. Scene file: top level

```json
{
  "p2s_version": 1,
  "name": "tank_truck_crash",
  "meta": {"iteration": 1, "max_iterations": 8, "prompt": "A military truck crashes into ..."},
  "clock": {"fps": 24, "duration_s": 6, "substeps": 20, "preroll_s": 0},
  "world": {"gravity": [0, 0, -9.81], "ground": {"material": "asphalt"}, "seed": 0,
            "wind": {"velocity": [7, 0, 0], "gust": 0.2, "turbulence": 0.3, "turbulence_length_m": 1.0},
            "water": {"level": 0, "density": 1025, "current": [0, 0.2, 0], "min": [x, y], "max": [x, y]}},
  "materials": [...], "bodies": [...], "multirotors": [...], "walkers": [...], "signals": [...],
  "vehicles": [...], "tracked_vehicles": [...], "joints": [...], "soft_bodies": [...], "cloths": [...],
  "particles": [...], "magnets": [...], "explosions": [...], "triggers": [...], "refine": [...], "cameras": [...], "expect": [...], "acknowledge": []
}
```

- `clock.duration_s * fps + 1` frames are recorded (6 s at 24 fps = 145 frames; video models like `1 + 4n`).
- `clock.substeps` physics steps per frame (20 at 24 fps = 480 Hz). Raise it for stiff or fast scenes.
- `clock.preroll_s` simulates before t = 0 without recording (a flag settling in the wind).
- `world.ground`: `true` (default), `false` (no ground: space, or a scene with its own floor), or
  `{"material", "height"}`.
- `world.water`: a flat water surface at `level` (optional horizontal `min`/`max`; unbounded without). Every
  dynamic body then floats or sinks by its real density: buoyancy acts on the submerged part of each shape, so
  a hull rights itself, and water drag (`hull_cd` on the body, 0.8) slows it. Put `ground.height` below it as the
  sea bed. Water here is a field, not particles: use `particles` for water that pours or splashes.
- `refine`: smaller steps while two rigid bodies are close: `{"between": ["truck", "tank"], "within_m": 1.0,
  "hold_s": 0.4, "factor": 8}`. Use it for every fast collision (it is what made the tank-truck crash converge).

## 5. Shapes and materials

Every collision shape is an exact primitive; the checker and the renderer use the same shapes, so **model what
should be seen and touched**: a truck is a cab box, a bed box, a front-end prism and wheels, not one box.

| type | keys | notes |
|---|---|---|
| `box` | `size` [x, y, z] or `half_extents` | |
| `sphere` | `radius` | |
| `capsule` | `radius`, `half_length` | axis along local Z, `half_length` excludes the caps |
| `cylinder` | `radius`, `half_length` | axis along local Z (rotate with `roll_deg: 90` for a wheel or a log) |
| `prism` | `profile` [[x, y], ...] convex, counter-clockwise; `z0`, `z1` | an outline extruded along Z (car noses, wedges, panels) |
| `convex` | `points` [[x, y, z], ...] | convex hull of the points |
| `mesh` | `vertices`, `triangles` | static bodies only (terrain, ramps) |
| `container` | `radius`, `height`, `wall` (0.006), `base` (0.01), `segments` (16) | open-topped cup, glass, bucket, tank; origin at the bottom centre |

Each shape also takes `pos`, an orientation (local to the body) and `material`.

Materials: presets `default steel aluminium rubber tyre wood concrete asphalt ice glass plastic ceramic flesh track
cloth`, or define your own: `{"id": "mud", "preset": "concrete", "static_friction": 0.4, "dynamic_friction": 0.3,
"restitution": 0.0, "density": 1800}`. A body without `mass_kg` or `density` weighs what its shapes' materials say.

## 6. Entities

### 6.1 `bodies`: rigid bodies

```json
{"id": "crate", "type": "dynamic", "shapes": [{"type": "box", "size": [1, 1, 1], "material": "wood"}],
 "pos": [0, 0, 0.5], "yaw_deg": 30, "mass_kg": 40,
 "velocity": [0, 0, 0], "angular_velocity": [0, 0, 0],
 "drag": {"cd": 1.05, "area_m2": 1.0}, "damping": {"linear": 0, "angular": 0.05},
 "ccd": false, "assembly": "", "solver_iterations": 8, "max_depenetration_mps": 2.0}
```

- `type`: `dynamic` (forces and contacts move it), `kinematic` (follows `keys`, pushes others, is not pushed
  back), `static` (never moves: buildings, poles, tables, walls).
- Kinematic path: `"keys": [{"t": 0, "pos": [...], "yaw_deg": 0}, {"t": 3, "pos": [...]}]`, eased between keys.
  Use kinematic only for things moved by something outside the simulation. Things that fall, roll, slide,
  collide or are pushed must be dynamic: **do not keyframe outcomes**.
- `drag`: `linear_per_s` (simple damping, e.g. 0.3 for a parcel under a drone) or `cd` + `area_m2` (quadratic,
  uses the wind). Spinning bodies keep gyroscopic effects (tops, thrown wrenches, tumbling debris).
- `ccd`: continuous collision for small fast bodies (bullets, balls, rockets); r warns about anything over
  150 m/s without it. `assembly`: bodies with the same assembly never collide with each other.
- `thrusters`: forces fixed to the body: rocket motors, jets, ship propellers, bow thrusters.
  `{"id": "motor", "pos": [0, 0, -0.6], "direction": [0, 0, 1], "force_n": 600, "burn_s": [0.5, 2.5]}`, in the
  body frame; or a propeller driven by a spinning revolute joint: `"joint": "prop_shaft", "k": 2.5` gives thrust
  k w|w|. `"medium": "water"` makes it push only while under water (a ship's screw), `"air"` only above it;
  `"active": false` waits for the verb `ignite`.
- Verbs (section 6.11): `impulse {vector, point?}` (N s), `push {force, duration_s}` (N), `set_velocity
  {velocity}`, `release` (kinematic becomes dynamic: drop it, fire it), `freeze`, `ignite`/`cutoff {thruster}`.
- A bullet or rocket that starts from rest in a launcher: make it `kinematic`, then at the firing time trigger
  `release` and (bullet) `set_velocity`. A light projectile can only move what its momentum allows: a 10 g
  bullet at 850 m/s barely nudges a 16 kg block but knocks a 60 g tin can away.

### 6.2 `joints`

```json
{"id": "hinge", "type": "revolute", "a": "frame", "b": "door", "anchor": [1, 0, 1], "axis": [0, 0, 1],
 "limits": {"lower": -90, "upper": 90}, "motor": {"velocity": 1.0, "max_force": 50},
 "break_force": 5e4, "break_torque": 1e4}
```

- `type`: `fixed` (glue; a carried payload), `revolute` (hinge; `limits` in degrees, optional `motor` in rad/s and
  N m), `prismatic` (slider; `limits` in metres), `spherical` (ball joint), `distance` (rope or spring:
  `min_distance`, `max_distance`, `spring: {stiffness, damping}`).
- `a` may be `"world"`. `anchor` and `axis` are world coordinates at t = 0.
- `break_force` / `break_torque`: the joint snaps beyond these (event `<id>:broken`): collapsing structures,
  towing ropes that snap.
- `wind_rotor` (revolute): `{"radius_m": 17, "cp": 0.45, "tip_speed_ratio": 7}` lets the wind turn `b` about the
  axis (wind turbines, windmills, pinwheels): rated power cp rho/2 pi R^2 v^3 at the design tip speed ratio, a
  slow spin-up for big rotors. Point `axis` into the wind.
- Rotors of every kind are revolute joints: a ceiling fan or ship propeller is a `motor` (rad/s, N m), a wind
  turbine a `wind_rotor`; drones are `multirotors` (6.3). Check the rate with a `spin` expectation.
- Verbs: `release` (detach now: a drone drops its parcel), `motor {velocity, max_force}`.

### 6.3 `multirotors`: drones flown by their rotors

All rigid-body keys (`shapes`, `pos`, `mass_kg`, `drag`) plus:

```json
"rotors": {"layout": "quad_x", "arm_m": 0.25, "height_m": 0.05},
"rotor": {"thrust_to_weight": 2.0, "max_rpm": 9000, "km_over_kf_m": 0.016, "tau_s": 0.03},
"controller": {"goals": [{"from_t": 0.5, "t": 3.0, "pos": [0, 0, 5]}, {"t": 6.5, "pos": [8, 3, 6]}],
               "yaw_deg": 0, "tilt_max_deg": 35, "position_bandwidth": 2.2, "attitude_bandwidth": 25}
```

- Layouts: `quad_x quad_plus hex octo`, or explicit `[{"pos": [x, y, z], "spin": 1}, ...]`.
- Each rotor's thrust is kf w^2 along the body's up axis; the controller tilts the craft to fly, so it leans into
  its motion, sags under payloads and fights gusts like a real drone.
- A goal is reached at `t` from wherever the drone is at `from_t` (default: the previous goal's `t`). A drone
  resting on the ground keeps its motors off until the first goal begins (event `<id>:lift_off`).
- Carry a parcel: a dynamic body under it plus a `fixed` or short `distance` joint, and a trigger `release`.
- Verbs: rigid-body verbs plus `kill` (motors off: it falls) and `goto {pos, t_s}`.

### 6.4 `vehicles`: wheeled vehicles with a driver

```json
{"id": "car", "pos": [0, 0, 0], "yaw_deg": 90, "speed_mps": 10, "mass_kg": 1500,
 "shapes": [{"type": "box", "size": [4.4, 1.8, 0.7], "pos": [0, 0, 0.75]},
            {"type": "box", "size": [2.2, 1.6, 0.5], "pos": [-0.3, 0, 1.35]}],
 "wheels": {"radius_m": 0.33, "half_width_m": 0.11, "mass_kg": 20,
            "axles": [{"x": 1.35, "track_m": 1.55, "steer": true, "drive": true},
                      {"x": -1.35, "track_m": 1.55, "steer": false, "drive": false}]},
 "suspension": {"travel_m": 0.2, "frequency_hz": 1.6, "damping_ratio": 0.3}, "tyre_friction": 0.9,
 "limits": {"max_wheel_torque_nm": 1200, "max_brake_torque_nm": 3000, "max_steer_deg": 32},
 "driver": {"path": [[0, 0], [0, 60], [5, 75]], "cruise_mps": 12, "accel_max": 2.0, "brake_comfort": 2.0,
            "brake_max": 6.0, "lat_accel_max": 2.5, "headway_s": 1.2, "gap_min_m": 2.0,
            "leader": "car_ahead", "signal": "tl_ns", "stop_s": 40, "yield_to": ["ped_a"],
            "crosswalks": [[42, 46]], "lane_width_m": 3.5}}
```

- Chassis shapes are in the body frame (origin on the ground at the vehicle's centre, +X forward). Keep them above
  the tyre tops (z > 2 * radius) where they overlap the wheels in plan view. Wheels and chassis never collide with
  each other; wheels ride the road on their suspension and collide with everything else.
- The driver steers along `path` (world [x, y] points; arc lengths along it are `stop_s` and `crosswalks`),
  slows for bends, follows its `leader`, stops for red (and for yellow when it can stop at `brake_yellow`), and
  waits while a `yield_to` walker is on a crosswalk in its lane. Events: `<id>:stops`, `<id>:moves`,
  `<id>:passes_stop_line` (with the light's colour).
- List axles front to back. Typical numbers: car 1.2-2 t with 800-1500 N m per driven wheel and 2500-3500 N m of
  brake per wheel; truck 7-15 t with 3000 N m and 6000 N m; r reports the result in W/kg and g so you can compare.
- Verbs: `brake {level 0..1}` (the driver stamps on the brake: after a crash), `release` (lets go of everything),
  `set_speed {mps}`.
- A driver without `signal` ignores the lights (it runs a red); one without `leader` ignores other vehicles and
  drives into them: that is how a crash is staged.
- `com` [x, y, z] (body frame) sets the centre of mass. A van or truck hit side-on rolls over unless it is low
  (about a third of the body's height); lower it if r shows wheels `below_ground` after a side impact.
- `substeps`: 20 at 24 fps suits cars; crash precision comes from a `refine` window (factor 8). Raising substeps and
  the refine factor together makes the wheel suspension stiff and twitchy; change one at a time.

### 6.5 `tracked_vehicles`: tanks, bulldozers

All rigid-body keys plus `"traction": {"speed_mps": 5, "mu": 0.7, "track_half_length_m": 1.8, "response_s": 0.25,
"path": [[x, y], ...], "lat_accel_max": 2.0}`. Use the `track` material on the track boxes (they slide; the traction
controller supplies grip up to `mu * m * g`). Without `path` it holds its heading; with one it skid-steers along it
and slows for bends. Verbs: rigid-body verbs plus `brake` and `set_speed {mps}`.

### 6.5b `robot_arms`: robot arms that carry things

```json
{"id": "robot", "robot": "franka_panda", "base": {"pos": [-0.1, 0, 0.75], "yaw_deg": 0},
 "hold": {"object": "pitcher", "offset": {"pos": [0.13, 0, 0], "pitch_deg": -90}, "gripper_m": 0.035},
 "point_local": [0.05, 0, 0.16],
 "keys": [{"t": 0, "point": [0.42, 0.143, 1.06], "yaw_deg": -90, "tilt_deg": 0},
          {"t": 2.3, "point": [0.5, -0.03, 0.94], "yaw_deg": -90, "tilt_deg": 45}],
 "drive": {"stiffness": 2000, "damping": 200, "max_torque_nm": 87}}
```

- Robots: `franka_panda` (made from MuJoCo Menagerie by `sim/physx/tools/mjcf_to_robot.py`; others can be added the
  same way). The arm is a jointed physical body: its joints are driven to an inverse-kinematics solution of the
  keys, so it moves, carries and pushes like a real arm.
- `keys` say where a point of the held object (`point_local`, in the object's frame) should be, which way the object
  faces (`yaw_deg`) and how far it is tilted about its own Y axis (`tilt_deg`; for a pitcher with its spout on +X,
  positive tilts the spout down). Without `hold`, the keys move the tool centre point.
- The held object is a dynamic body (a `container` pitcher, a box); the arm places it in the gripper at t = 0.
  Put things the arm stands on (the table) in the arm's assembly: `"assembly": "<arm id>"`.
- A camera can ride a link: `{"type": "attached", "parent": "robot", "part": "hand", ...}`.

### 6.6 `walkers`: people

```json
{"id": "ped_a", "path": [[-8, 4], [8, 4]], "speed_mps": 1.4, "start_s": 1.0, "start_offset_m": 0}
{"id": "bystander", "idle_at": [5, -6], "facing_deg": 90, "pose": "phone", "sway_m": 0.02}
```

Kinematic capsules (`height_m` 1.75, `radius_m` 0.25): they push things and are never pushed. The renderer poses a
human from their walk phase.

### 6.7 `signals`: traffic lights

`{"id": "tl_ns", "plan": [{"state": "green", "until_s": 3}, {"state": "yellow", "until_s": 4}, {"state": "red",
"until_s": 30}], "heads": [{"pos": [6, 8, 5], "yaw_deg": 180}]}`. Events `<id>:<state>` at each change.

### 6.8 `soft_bodies`: deformable solids (elastic or permanently crushed)

```json
{"id": "truck_front", "host": "truck",
 "shape": {"type": "prism", "profile": [[2.42, 1.25], [3.58, 1.25], [3.58, 0.8], [3.8, 0.8], [3.8, 2.0], [2.42, 2.0]],
           "z0": -1.32, "z1": 1.32, "roll_deg": 90},
 "mass_kg": 1500, "youngs_pa": 2e7, "poisson": 0.3,
 "plastic": {"yield_strain": 0.001, "low_limit": 0.2, "high_limit": 5.0},
 "attach": [{"min": [2.3, -1.5, 1.0], "max": [2.5, 1.5, 2.1]}],
 "resolution": 20, "max_edge_m": 0.12, "solver_iterations": 30}
```

- With a `host`, the shape and the `attach` boxes are in the host's frame and the attached vertices are bolted to
  it; without a host, `pos` places it in the world and `attach` boxes pin vertices to the world. A mounted soft body
  never collides with its host.
- Soft-body prisms may be concave (an L-shaped bumper over a wheel well); rigid prisms must be convex.
- Elastic: rubber balls, jelly, cushions, a car tyre. Plastic (`plastic`): crumple zones, dented panels, bent
  metal, dough. Stiffness guide: rubber 1e6-1e7, crumple zone 2e7 (crushes ~0.3 m at 16 m/s), armour 5e8.
- A soft body reports contacts like a rigid one and also as its host (`contact:tank:truck` fires when the truck's
  crumple zone touches the tank's armour).
- Mesh density: `max_edge_m` sets the collision mesh (what r counts in `fem_inverted` and draws); `resolution` sets
  the voxel grid the solver uses. Halving `max_edge_m` roughly quadruples the tetrahedra.
- Size a crumple zone to the impact patch: `deformation` is the 90th percentile over all its vertices, so a whole
  van side hit by a car nose reads as almost no deformation. Use a panel about as wide as what hits it.
- Attach only the face that touches the host (a thin slab of the `attach` box); attaching most of the body makes it
  rigid.
- A crumple zone must hold its shape until something hits it. r fails `deformed_early` if it sags or flops before any
  contact (too soft for its mass and the host's accelerations): raise `youngs_pa` and `yield_strain`.
- `fem_inverted`: a few percent of inverted tetrahedra at a crush face is normal for a hard crash (warning from
  0.5%; acknowledge it if the deformation numbers look right); above 10% the solve failed (failure).

### 6.9 `cloths`: flags, banners, tarps, sheets

```json
{"id": "flag", "size_m": [2.4, 1.6], "grid": [41, 28], "pos": [0.07, 0, 6.25], "roll_deg": 90,
 "host": "pole", "pin": [{"min": [0, 0], "max": [0.001, 1.6]}],
 "areal_density": 0.25, "thickness_m": 0.002, "youngs_pa": 5e6, "bending": 2e-6,
 "drag": {"normal": 1.2, "tangent": 0.15}}
```

The rectangle lies in its local x-y plane from (0, 0) to `size_m`; `pin` boxes are in that plane. Wind comes from
`world.wind`. Use `clock.preroll_s` (1-2 s) so a flag is already flying at t = 0.

### 6.10 `particles`: liquids, sand, smoke

```json
{"id": "oil", "preset": "oil", "spacing_m": 0.012, "max_particles": 30000,
 "blocks": [{"min": [-0.1, -0.1, 0.3], "max": [0.1, 0.1, 0.5]}],
 "emitters": [{"host": "truck", "pos": [3.1, 1.38, 0.85], "direction": [0, 0, -1], "radius_m": 0.055,
               "speed_mps": 1.6, "active": false, "phases": [{"after_s": 0.5, "radius_m": 0.022, "speed_mps": 1.0}]}]}
```

- Presets: `water oil honey` (kind fluid), `sand` (granular), `smoke`. Override `density viscosity cohesion
  adhesion surface_tension friction damping`; smoke also `buoyancy` (fraction of g, 0.12), `lifetime_s` (4),
  `drag` (1/s toward the wind, 1.0), `diffusion_mps` (random mixing that widens the plume, 0.4).
- `blocks` fill boxes at t = 0; `emitters` (on a `host` or in the world) shoot disc layers along `direction`;
  `phases` change the radius and speed over time after the emitter starts (a gush, then a trickle).
- Emitters with `"active": false` wait for a trigger `start` (verbs `start`, `stop`).
- Containers need walls at least two particle spacings thick (`container.wall` >= 0.025 for 1.2 cm particles).
- There is no fire: smoke from a burning thing is a `smoke` emitter; the video model draws the flames.

### 6.10b `magnets`

```json
{"id": "crane_magnet", "host": "crane_head", "pos": [0, 0, -0.2], "targets": ["plate"],
 "force_n_at_1m": 2500, "max_force_n": 6000, "falloff": 2, "active": false}
```

Pulls (or with `"repel": true` pushes) each target body: force_n_at_1m / d^falloff from the magnet point to the
target's nearest point, capped at `max_force_n`; the host (or the world) feels the opposite force. Only list iron
and steel targets. Verbs: `on`, `off`, `repel`, `attract`.

### 6.10c `explosions`

```json
{"id": "blast", "pos": [0, 0, 0.1], "at_s": 1.0, "radius_m": 8, "impulse_ns": 700, "particle_speed_mps": 6}
```

Only the effect on the surroundings is simulated: at `at_s` (or on the verb `detonate`) every dynamic body within
`radius_m` gets an impulse away from the centre, `impulse_ns` x its facing area / max(d, 1 m)^2, at its nearest
point (so it tumbles); particles (sand, water, smoke) are thrown at up to `particle_speed_mps`. Event
`<id>:detonated`. No fireball or flash: add a smoke emitter started by the event, the video model does the rest.
Scale: a few hundred N s for a gas cylinder flinging crates metres, thousands for a car bomb.

### 6.11 `triggers`: things that happen on cue

```json
{"on_event": "contact:tank:truck", "delay_s": 0.6, "target": "truck", "do": "brake", "params": {"level": 1}}
{"at_s": 5.0, "target": "hook", "do": "release"}
{"on_event": "contact:tank:truck", "delay_s": 0.06, "target": "oil", "do": "start"}
```

Each trigger fires once. `do` must be a verb of the target's kind:

| kind | verbs |
|---|---|
| rigid body, multirotor, tracked vehicle | `impulse`, `push`, `set_velocity`, `release`, `freeze`, `ignite`, `cutoff` (+ multirotor `kill`, `goto`; tracked `brake`, `set_speed`) |
| vehicle | `brake`, `release`, `set_speed` |
| joint | `release`, `motor` |
| particles | `start`, `stop` |
| cloth | `release` (unpin: a flag tears off) |
| magnet | `on`, `off`, `repel`, `attract` |
| explosion | `detonate` |

### 6.12 `cameras`

```json
{"id": "cctv", "type": "fixed", "position": [12, -10, 6], "look_at": [0, 0, 1], "fov_deg": 70, "resolution": [1280, 720]}
{"id": "dashcam", "type": "attached", "parent": "car", "offset": {"pos": [1.8, 0, 1.3], "pitch_deg": -3}}
{"id": "phone", "type": "handheld", "position": [6, -8, 1.6], "look_at_entity": "truck", "shake": {"rot_deg": 0.6, "pos_m": 0.004, "seed": 11}}
{"id": "drone_cam", "type": "orbit", "center_entity": "tank", "radius_m": 25, "height_m": 15, "deg_per_s": 12, "start_deg": 0}
{"id": "chase", "type": "follow", "target": "truck", "offset": [-12, 0, 5], "lag_s": 0.3}
{"id": "satellite", "type": "fixed", "position": [0, 0, 400], "look_at": [0, 0.01, 0], "fov_deg": 8}
```

- Attached cameras look along their parent's +X with its +Z up; `offset` moves and turns them in the parent's frame:
  `yaw_deg` > 0 turns left, **`pitch_deg` > 0 tilts the view down** (so a dashcam looking slightly up has
  `pitch_deg: -4`), `roll_deg` rolls. `part` picks a part of the parent (a robot link: `"part": "hand"`).
- Handheld cameras may ride a walker (`parent`, `offset.pos`) and still aim at `look_at` or `look_at_entity`.
- A camera's own parent never blocks its view in the checker (a dashcam sees out of its car).
- `satellite`-style views: a high fixed camera with a narrow `fov_deg` and `clip_m` that reaches the ground.

## 7. `expect`: what the prompt asks for

The checker can only confirm what you state. Write one expectation per event, outcome and shot in the prompt; r
fails with `expect_coverage` if a camera has none or there is no physical outcome.

| type | keys | passes when |
|---|---|---|
| `event` | `event`, `within_s` [a, b] | the event first happens in the window |
| `no_event` | `event` | it never happens (no contact between the car and a pedestrian) |
| `event_order` | `events` [...] | first occurrences are in this order |
| `in_view` | `camera`, `entity`, `during_s`, `min_fraction` 0.9, `min_screen_fraction` 0, `max_occluded` 0.5 | the entity is visible enough in that share of frames |
| `enters_view` | `camera`, `entity`, `outside_at_s`, `inside_by_s` | out of view at the first time, in view by the second |
| `leaves_view` | `camera`, `entity`, `inside_at_s`, `outside_by_s` | the reverse |
| `event_in_view` | `camera`, `event` | where the event happens is inside the frame when it happens |
| `reaches` | `entity`, `point`, `within_m`, `by_s` | it gets that close by then |
| `speed` | `entity`, `during_s`, `min_mps`, `max_mps` | speed stays in range |
| `at_rest` | `entity`, `after_s`, `max_speed_mps` 0.05 | it has stopped |
| `min_gap` | `a`, `b`, `metres` | they never come closer |
| `upright` | `entity`, `max_tilt_deg` 30 | it never tips further |
| `spin` | `entity`, `during_s`, `min_rpm`, `max_rpm` | its rotation rate stays in range (rotors, wheels, tops) |
| `turns` | `entity`, `during_s`, `min_deg`, `max_deg` | its heading changes that much in total (a car spinning out, a door swinging open) |
| `deformation` | `entity`, `min_m`, `max_m` | a soft body's permanent crush (90th percentile) is in range |
| `particles_in_region` | `entity`, `min`, `max`, `at_s`, `min_fraction` 0.8 | that share of particles is in the box |
| `obeys_signals` | `vehicles` (optional list) | no vehicle (or none of those listed: leave out the one that runs the red) crosses its stop line on red |

Give each expectation a unique `id`; findings are keyed `expect_<type>:<id>`.

- `within_s` is optional: without it an `event` only has to happen. Events during `clock.preroll_s` have negative
  times, and a contact that began in the preroll (a tank resting on the ground) is not a new contact after t = 0.
- `min_gap` measures surface to surface (a body includes the soft bodies mounted on it) and counts a contact as 0.
- Ids that name no entity or camera, and contact events with ids out of alphabetical order, are errors.
- In camera checks only rigid bodies block the view; cloth, soft bodies, smoke and liquids never hide anything.
  Visibility counts only the side of an entity facing the camera.
- `spin` above half a turn per frame (360 rpm at 24 fps) cannot show in video; r says so.

## 8. What r checks besides your expectations

- **Hard checks:** finite state, no PhysX errors, no interpenetration beyond about 3 cm (bodies, deformables,
  particles; overlap at t = 0 is reported separately), nothing below the ground or flung away, no energy gained
  beyond what actuators delivered, no inverted FEM elements.
- **Plausibility tables for you to judge:** per body, peak speed and peak acceleration |dv/dt| in g (0 at rest,
  1 g in free fall) and whether a contact explains it; per actuator, thrust-to-weight, W/kg and braking g against the usual ranges. Out-of-range values are
  warnings: fix them, or acknowledge them when the prompt really asks for it.
- **Cameras, from geometry alone:** where each image's corner rays meet the ground (what the camera covers), and
  for every moving entity and every entity an expectation names: the share of frames it is visible, how much of it
  is unoccluded, its screen size and an in/out timeline; static scenery gets one summary line per camera.
- Failures are listed in the order to fix them (simulation and overlap first, expectations last).
- **Fact card:** measured events, start and end positions, peak speeds, contacts with peak forces, deformation.
  Use these numbers, not your intentions, when describing the scene later.

## 9. Modelling guide

- Make it physical: give things real masses, speeds and materials, then let the solver produce the outcome. Use
  controllers (drivers, rotors, traction, motors) for agents, triggers for decisions ("brakes 0.6 s after the
  hit"), and kinematic keys only for people and props moved by hand.
- Plan positions from the desired timing: to hit at t = 1.5 s, start the truck `v * 1.5` m from the contact point.
- Fast impacts (> 5 m/s): add a `refine` window for the pair; for crushing, mount `soft_bodies` with `plastic` on
  the colliding fronts (the truck's crumple zone, the tank's armour panel) and keep rigid shapes clear of them.
- Pouring: a `container` held in a gripper or kinematic `keys`, liquid as `particles` in it, a receiving
  `container` below; expect `particles_in_region`.
- Wind effects: `world.wind` plus `cloths` (flags), `particles` smoke, and `drag` with `cd` on light bodies.
- Boats: a hull box (or prism) with the real mass, a propeller body on a motored revolute joint, and a
  `thrusters` entry linked to that joint with `"medium": "water"`. Submarines: mass = displaced volume x 1025 (neutral),
  a water thruster. Rockets: `thrusters` with `burn_s`, `ccd`, held `kinematic` on the pad until `release`.
- Not simulated (leave to the video model): flames and fireballs, thin stiff sheets that bend and spring back
  (playing cards, sheet metal flexing), cutting and fracture, liquid-surface waves and wakes.
- Many similar bodies (traffic, debris, crates): fine, up to thousands; keep ids short and systematic (`car_01`).
- Cameras: point each one at where the action is when it happens; the coverage tables in r tell you if you missed.

## 10. Example

`sim/physx/scenes/` holds working scenes that pass the checker. Start from the closest one:

| scene | shows |
|---|---|
| `tank_truck.json` | wheeled truck with a plastic crumple zone, tracked tank with plastic armour, crash refine window, oil emitter started by the impact, flag in gusty wind, five cameras incl. satellite, phone and orbiting drone |
| `intersection.json` | four cars with drivers, two signals, pedestrians on crosswalks, buildings; CCTV, dashcam, phone |
| `robot_pour.json` | robot arm pouring water from a pitcher into a glass; wrist camera on the hand link |
| `drone_takeoff.json`, `drones_air.json` | drones lifting off, crossing, hovering in gusts |
| `convoy_overhead.json` | tracked tank and two trucks following a road under a bridge and a tree canopy; overhead view |
| `smoke_and_sand.json` | smoke drifting downwind, a crate falling into sand, a spinning top |
| `harbour.json` | motor boat driven by a propeller joint, submarine with a water thruster, ballasted buoy; `world.water` |
| `blast_yard.json` | explosion throwing crates and a drum, smoke started by the blast, crane electromagnet lifting a plate |
| `range.json` | rocket held on a rail, released at ignition; bullet (`ccd`) knocking a can off a stack |
| `rotors.json` | wind turbine turned by the wind (`wind_rotor`), motored ceiling fan; `spin` expectations |
