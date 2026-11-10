# Proof of Concept: Drones Crossing in the Air (Nov 10, 2026)

**Owner:** SP Sharan · **Status:** findings from a one-day proof of concept run before the full [v0 engineering plan](2026-11-10-0012-v0-engineering-plan.md). One scene, one seed per condition, judged by eye with no metrics. Treat every result as a first look, not a measurement.

**Where to look:**
- Grid video: `/home/ss99569/code/sim2real-videogen/data/poc/drones_air/grid.mp4` on swarm5 (contact sheet: `grid_frames.png`)
- Agent's run notes, with per-clip cost and observations: `/home/ss99569/code/sim2real-videogen/data/poc/drones_air/notes.md`
- Code: `sim/`, `gen/`, `configs/` on `spsharan/v0-pipeline`; how to re-run it: [setup](../setup/README.md#re-run-the-drones_air-proof-of-concept)
- Problems hit and their fixes: [Step 1 gotchas doc](2026-11-10-0945-step1-environment-gotchas.md)

## 1. Bottom line

- **Controls work and text doesn't.** Every condition given a simulator control kept both drones, followed the scripted path and kept identity through the crossing. Conditions given only text, or text plus the first frame, never produced the crossing, even when the path was written out in detail.
- **Sparse boxes were enough.** VACE given only box outlines followed the path as well as VACE given dense depth. That's a dense vs. sparse comparison on one model, which the lit review flagged as missing from the literature.
- **Realism depends on whether the model gets our render as frame 0.** Models given frame 0 copy the simulator's flat look for the whole clip. Models given only a control and text produce real-looking footage, but invent what the control doesn't specify (surfaces, colours).
- **Cost differs about 8×.** Cosmos Sim-Transfer takes about 3 min on half a card. VACE-14B and Wan-Move take about 27–31 min on a whole card.

## 2. Setup

**Scene (`drones_air`):** two procedural quadcopters (orange, nearer, ~7 m; white, farther, ~10 m) cross in front of a static camera, with sky, tree line and a grass field behind. 5.1 s, 153 frames at 30 fps, 832×480. Drones are 70–109 px wide and fully in frame the whole time. They swap left/right order at 2.63 s, with the orange one in front. Rendered with BlenderProc / Cycles.

**Prompt (same for every condition except A+):** "Two small quadcopters fly across the sky in front of a static camera, passing each other in mid-air. Behind them is a soft blue sky over a distant flat horizon. Realistic footage, smooth natural motion, the camera does not move." A+ adds a detailed description of who starts where, which way each moves, when they pass and where each ends (`configs/prompts/drones_air.yaml`).

**Conditions:** seed 0 throughout.

| | What the model gets | Model | Control type |
|---|---|---|---|
| R | — (raw simulator render) | Blender | reference |
| A | Text | VACE-14B | none |
| A+ | Detailed text describing the path | VACE-14B | none |
| B | Text + our render's frame 0 | VACE-14B | none |
| B-box | Frame 0 + simulator boxes drawn as outlines | VACE-14B | sparse |
| C | Frame 0 + depth | VACE-14B | dense |
| C0 | Depth + text, no frame 0 | VACE-14B | dense |
| D | Depth + text, no frame 0 | Cosmos3-Nano-Sim-Transfer | dense |
| E | Frame 0 + 32 point tracks | Wan-Move-14B | sparse |

VACE-14B ran with its weights cast to bf16 (it runs out of memory on an 80 GB A100 as shipped).

## 3. Results

| Cond | Both drones kept | Crossing, identity kept | Follows path | Look |
|---|---|---|---|---|
| A | No: one leaves by 1.5 s | No crossing | No | Real footage; red consumer drones, larger than in R |
| A+ | No: a third drone appears, one flies at the camera and leaves | No crossing | No | Real footage; ignores the stated left/right order and colours |
| B | Yes | No crossing: both hover and drift | No | Simulator look |
| B-box | Yes | Yes | Yes | Simulator look; drone shapes deform by the end |
| C | Yes | Yes | Yes, close to the simulator | Simulator look |
| C0 | Yes | Yes | Yes | Real footage; the field becomes a lake, drones become red and yellow |
| D | Yes | Yes | Yes | Real footage; drones turn black, new scene (dirt field, autumn trees) |
| E | Yes | Yes | Yes | Simulator look; slight colour tint |

## 4. Findings

### 4.1 Text can't direct the event, even spelled out

- A (text) and B (text + first frame) never produce the crossing. A loses a drone; B keeps both, but they only hover.
- A+ was the test of whether writing the path down is enough without a simulator. It isn't: VACE ignored the left/right order, added a third drone and flew one drone into the camera.
- This is the clearest argument for using a simulator at all: the event we want can't be had from the prompt.

### 4.2 Every control worked, dense and sparse alike

- Dense depth (C, C0, D) and sparse input (B-box boxes, E point tracks) all kept both drones, followed the path and kept identity through the crossing.
- **B-box matters most here.** It's on the same VACE checkpoint as C, so C vs. B-box isolates dense vs. sparse without a model change. On this scene, boxes were enough.
  - Both boxes are drawn in the same colour (VACE colours by category), so identity comes only from frame 0, not from the boxes.
  - The drone shapes deform by the end of the clip, so sparse control holds position better than appearance.
- E (Wan-Move tracks) is also sparse but on a different model, so E vs. C mixes the input change with a model change.

### 4.3 Realism depends on frame 0, not on the model

- B, B-box, C and E all get our render as frame 0, and all keep the simulator's look for the whole clip: flat-shaded drones, toy trees, pale sky. No sim-to-real happens.
- C0 and D get the same kind of input (depth + text, no frame 0) on two different models, and both produce real-looking footage. So VACE can make realistic video from our simulator; it just copies our render when given one.
- **What the control doesn't say, the model invents.** Depth carries position and shape, not material or colour. C0 turned the grass field into a lake and recoloured the drones; D turned them black and changed the whole scene. Our prompt only said "distant flat horizon", so naming the ground and drone colours in the prompt would likely fix much of this (untested).
- **Implication for the pipeline:** a realistic first frame would let frame-0 conditions look real too. One option is to take the first frame from a no-frame-0 output (e.g. Cosmos or C0) and feed it to VACE or Wan-Move, using one model for appearance and another for motion. Untested.

### 4.4 Cost

| Cond | Model | Wall time | Peak GPU |
|---|---|---|---|
| R | Blender / Cycles | 5.3 min | 1.8 GB |
| A, A+, B, B-box, C, C0 | VACE-14B (bf16) | 26–28 min each | 74 GB |
| D | Cosmos3-Nano-Sim-Transfer | 3.3 min | 42.5 GB |
| E | Wan-Move-14B | 31.2 min | 40.8 GB |

- VACE-14B needs nearly a whole 80 GB card (74 GB peak).
- About 4 min of every VACE and Wan-Move run is prompt encoding on CPU. Prompts are frozen, so caching the encodings will save that in the full run.
- At these numbers, the full plan's VACE conditions (36 clips) need about 16 whole-card hours, more than the plan's estimate.

## 5. Caveats

- **One seed per condition.** The identity results at the crossing in particular need several seeds before we claim anything.
- **One easy scene.** Large drones, one crossing, clear sky, static camera. "Every control works" is close to circular here: when the model is handed the motion, it follows the motion. The interesting failures should show up in harder scenes (take-off, the overhead convoy, smaller drones) and across seeds.
- **Judged by eye.** No tracking or metrics yet.
- **Some differences are model differences, not input differences.** C vs. D and C vs. E compare different models; only A / A+ / B / B-box / C / C0 share one checkpoint.

## 6. Next steps

1. **The full plan:** all three scenes, 3+ seeds, the eval harness with tracking and audited labels ([engineering plan](2026-11-10-0012-v0-engineering-plan.md)).
2. **Add B-box and C0 to the condition matrix.** B-box gives the same-model sparse condition the plan lacked, and C0 is VACE's counterpart to Cosmos D.
3. **Name the ground and drone colours in the prompts,** and check whether C0 and D stop inventing surfaces.
4. **Try a realistic first frame** (from Cosmos or C0) for the frame-0 conditions.
5. **Model choice for the paper.** VACE is on Wan2.1 (early 2025). Two newer candidates that cover text, first frame and control from one checkpoint:
   - Wan2.2-VACE-Fun-A14B (Apache-2.0, same interface as VACE)
   - Cosmos3-Nano base, which we ran for depth transfer earlier, from the same family as D

   A short comparison on this scene's existing render and controls would settle it.
