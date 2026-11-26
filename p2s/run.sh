#!/usr/bin/env bash
# The whole pipeline for one experiment file, in one command. Every scene the file names is built from its prompt
# file; then every view is generated from every control the file lists, and every clip is scored.
#   source ~/.sim2real.env
#   bash p2s/run.sh p2s/experiments/v1/ltx25.yaml [gpu,gpu,...] [--until scene|render|generate]
#
#   stage      folder            what it does                                                    skips
#   1 scene    p2s/1_scene/      prompt file -> PhysX world, iterated by the SceneAgent to PASS   a world already accepted
#   2 render   p2s/2_render/     world -> one bundle per camera, with control videos             a bundle rendered from this world
#   3 generate p2s/3_generate/   captions + control video -> a clip per view, control and seed   a clip whose inputs are unchanged
#   4 score    p2s/4_score/      each clip against the simulator -> scores and one table         a clip already read
#
# The GPU list is for the Blender renders (one camera per GPU; default 0). Generation and scoring pick free GPUs
# themselves. Check nvidia-smi first: the server is shared. To stop generation: touch $S2R_DATA/<out_root>/STOP.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXP=${1:?usage: bash p2s/run.sh <experiment.yaml> [gpus] [--until scene|render|generate]}
GPUS=0; UNTIL=score
shift
while (( $# )); do
  case $1 in
    --until) UNTIL=$2; shift 2 ;;
    *) GPUS=$1; shift ;;
  esac
done
case $UNTIL in scene|render|generate|score) ;; *) echo "--until takes scene, render or generate"; exit 2 ;; esac
source "$REPO/env.sh"
cd "$REPO"
PY=$S2R_ROOT/envs/eval/.venv/bin/python
say() { printf '\n[p2s/run] %s\n' "$*"; }

# the experiment's scenes: name, prompt file (or "-" for a spec v2 scene of the older route), caption folder
mapfile -t SCENES < <("$PY" - "$EXP" <<'PYEOF'
import sys, yaml
exp = yaml.safe_load(open(sys.argv[1]))
for name, s in exp["scenes"].items():
    print(name, s.get("p2s", "-"), s["prompts"])
PYEOF
)
OUT_ROOT=$("$PY" -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['out_root'])" "$EXP")
NAME=$("$PY" -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['experiment'])" "$EXP")

for line in "${SCENES[@]}"; do
  read -r scene prompt captions <<< "$line"
  if [[ $prompt == - ]]; then
    say "$scene: a spec v2 scene of the older route; build it with legacy/sim/build_world.sh (not done here)"
    continue
  fi
  say "stage 1 (scene) + stage 2 (render): $scene from $prompt"
  if [[ $UNTIL == scene ]]; then
    "$PY" p2s/1_scene/write_scene.py "$prompt" --gpu "${GPUS%%,*}" ${P2S_WRITE_ARGS:-}
  else
    bash p2s/build_scene.sh "$prompt" "$GPUS"
  fi
  if [[ ! -f $captions/world.json ]]; then
    say "stage 3 (captions): $scene has no caption blocks; writing starter blocks to $captions"
    "$PY" p2s/3_generate/write_captions.py "$prompt" --out "$captions"
  fi
done
[[ $UNTIL == scene || $UNTIL == render ]] && { say "stopped after stage $UNTIL, as asked"; exit 0; }

say "stage 3 (generate): composing the prompts, then every job of $EXP"
"$PY" p2s/3_generate/compose_prompts.py --exp "$EXP"
"$PY" p2s/3_generate/run_multicam.py --exp "$EXP"
[[ $UNTIL == generate ]] && { say "stopped after stage generate, as asked"; exit 0; }

say "stage 4 (score): every finished clip of $EXP"
"$PY" p2s/4_score/run_eval.py --exp "$EXP"
mkdir -p "$S2R_DATA/$OUT_ROOT/tables"
"$PY" p2s/4_score/summarize.py --exp "$EXP" --by-scene --out "$S2R_DATA/$OUT_ROOT/tables/$NAME.csv"
say "done. Clips: $S2R_DATA/$OUT_ROOT/<scene>/r<rev>/gen/   Table: $S2R_DATA/$OUT_ROOT/tables/$NAME.csv"
