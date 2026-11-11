#!/usr/bin/env bash
# Wan2.1-VACE through diffusers: the same conditions and input files as gen/run_vacefun.sh, on the original VACE model.
#   SCENE=drones_air COND=Bbox_edit SEED=0 MODEL=14B bash gen/run_vace_diffusers.sh      # MODEL=14B (default) or 1.3B
# Every COND but A / Aplus reads src_video.mp4 and src_mask.mp4 from $S2R_DATA/poc/<scene>/<cond>/inputs/ (sim/controls.py).
# Output: $S2R_DATA/poc/<scene>/<cond>/vace21-<model>-s<steps>/seed<k>/out_video.mp4, run.json, run.log.
# VACE_DIFFUSERS_PY is a Python with torch and diffusers; VACE_DIFFUSERS_HUB is the HF cache holding
# Wan-AI/Wan2.1-VACE-<model>-diffusers. The GPU comes from the caller's environment.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCENE=${SCENE:?}; COND=${COND:?}; SEED=${SEED:-0}; STEPS=${STEPS:-30}; MODEL=${MODEL:-14B}
source "$REPO/env.sh"
PY=${VACE_DIFFUSERS_PY:?set VACE_DIFFUSERS_PY to a Python with torch and diffusers}
export HF_HUB_CACHE=${VACE_DIFFUSERS_HUB:?set VACE_DIFFUSERS_HUB to the HF cache with the diffusers weights} HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=$S2R_CACHE TORCH_HOME=$S2R_CACHE/torch MPLCONFIGDIR=$S2R_CACHE/matplotlib TOKENIZERS_PARALLELISM=false

KEY=base; [[ $COND == Aplus ]] && KEY=detailed
PROMPT=$("$PY" -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]])" "$REPO/configs/prompts/$SCENE.yaml" "$KEY")

IN=$S2R_DATA/poc/$SCENE/$COND/inputs
case "$COND" in
  A|Aplus) SRC=() ;;
  *) for f in "$IN/src_video.mp4" "$IN/src_mask.mp4"; do [[ -f $f ]] || { echo "missing $f" >&2; exit 1; }; done
     SRC=(--src_video "$IN/src_video.mp4" --src_mask "$IN/src_mask.mp4") ;;
esac

OUT=$S2R_DATA/poc/$SCENE/$COND/vace21-$MODEL-s$STEPS/seed$SEED
mkdir -p "$OUT"
echo "[vace21] scene=$SCENE cond=$COND seed=$SEED model=$MODEL steps=$STEPS CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
"$PY" "$REPO/gen/vace_diffusers_launch.py" --ckpt "Wan-AI/Wan2.1-VACE-$MODEL-diffusers" \
  --prompt "$PROMPT" --seed "$SEED" --steps "$STEPS" --save_dir "$OUT" "${SRC[@]}" 2>&1 | tee "$OUT/run.log"
