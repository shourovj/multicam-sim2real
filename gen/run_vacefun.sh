#!/usr/bin/env bash
# Wan2.2-VACE-Fun-A14B runner: the same conditions and input files as gen/run_vace.sh, on the newer model.
#   SCENE=drones_air COND=Bbox SEED=0 bash gen/run_vacefun.sh
# COND: A (text), Aplus (detailed text), B (text + frame 0), Bbox (frame 0 + boxes), C (frame 0 + depth), C0 (depth).
# Every COND but A / Aplus reads src_video.mp4 and src_mask.mp4 from $S2R_DATA/poc/<scene>/<cond>/inputs/ (sim/controls.py).
# Output: $S2R_DATA/poc/<scene>/<cond>/vacefun-a14b-s<steps>/seed<k>/out_video.mp4, run.json, run.log.
# STEPS (default 50) is the number of sampling steps; MEMORY_MODE (default model_cpu_offload) is VideoX-Fun's GPU memory mode.
# VIDEOX_FUN_DIR is the VideoX-Fun clone (with its .venv); the weights are linked under its models/ folder.
# The GPU comes from the caller's environment; nothing here touches CUDA_VISIBLE_DEVICES.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCENE=${SCENE:?}; COND=${COND:?}; SEED=${SEED:-0}; STEPS=${STEPS:-50}
source "$REPO/env.sh"
VIDEOX_FUN_DIR=${VIDEOX_FUN_DIR:-$S2R_ROOT/VideoX-Fun}
CKPT_DIR=${VACEFUN_CKPT:-$VIDEOX_FUN_DIR/models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B}
PY=$VIDEOX_FUN_DIR/.venv/bin/python

# Keep every cache except the HF one out of ~/.cache.
export XDG_CACHE_HOME=$S2R_CACHE TORCH_HOME=$S2R_CACHE/torch MPLCONFIGDIR=$S2R_CACHE/matplotlib TOKENIZERS_PARALLELISM=false

KEY=base; [[ $COND == Aplus ]] && KEY=detailed
PROMPT=$("$PY" -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]])" "$REPO/configs/prompts/$SCENE.yaml" "$KEY")

IN=$S2R_DATA/poc/$SCENE/$COND/inputs
case "$COND" in
  A|Aplus) SRC=() ;;
  *) for f in "$IN/src_video.mp4" "$IN/src_mask.mp4"; do [[ -f $f ]] || { echo "missing $f" >&2; exit 1; }; done
     SRC=(--src_video "$IN/src_video.mp4" --src_mask "$IN/src_mask.mp4") ;;
esac

OUT=$S2R_DATA/poc/$SCENE/$COND/vacefun-a14b-s$STEPS/seed$SEED
mkdir -p "$OUT"
echo "[vacefun] scene=$SCENE cond=$COND seed=$SEED steps=$STEPS CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
"$PY" "$REPO/gen/vacefun_launch.py" --videox_dir "$VIDEOX_FUN_DIR" --ckpt_dir "$CKPT_DIR" \
  --prompt "$PROMPT" --seed "$SEED" --steps "$STEPS" --memory_mode "${MEMORY_MODE:-model_cpu_offload}" \
  --save_dir "$OUT" "${SRC[@]}" 2>&1 | tee "$OUT/run.log"
