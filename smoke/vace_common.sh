# Shared settings for the VACE smoke tests. Sourced by smoke/vace_*.sh; not run directly.
#
#   MODEL=14B (default) or MODEL=1.3B
#   SEED   (default 2025, VACE's own default)
#   STEPS  (default 50, VACE's default for Wan2.1-VACE)
#   VACE_DIT_BF16=1  cast the DiT weights to bf16 after loading (see smoke/vace_launch.py)
#
# The GPU comes from the caller's environment; nothing here touches CUDA_VISIBLE_DEVICES.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
ROOT=$S2R_ROOT
VACE_DIR=$ROOT/VACE
PY=$VACE_DIR/.venv/bin/python
SMOKE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT_ROOT=${OUT_ROOT:-$S2R_DATA/smoke/vace}

MODEL=${MODEL:-14B}
SEED=${SEED:-2025}
STEPS=${STEPS:-50}

case "$MODEL" in
  14B|1.3B) ;;
  *) echo "MODEL must be 14B or 1.3B, got '$MODEL'" >&2; exit 1 ;;
esac
CKPT_DIR=$VACE_DIR/models/Wan2.1-VACE-$MODEL   # symlinks into the HF cache snapshots

# Keep every cache except the HF one out of ~/.cache.
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export XDG_CACHE_HOME=$S2R_CACHE
export TORCH_HOME=$S2R_CACHE/torch
export MPLCONFIGDIR=$S2R_CACHE/matplotlib
export UV_CACHE_DIR=$S2R_CACHE/uv
export TOKENIZERS_PARALLELISM=false

# Single-GPU memory savers: DiT/VAE offloaded to CPU between stages, T5 kept on CPU.
MEM_FLAGS=(--offload_model True --t5_cpu)

# 832 wide x 480 tall. VACE's SIZE_CONFIGS are (W, H); '480p' maps to (480, 832), which is portrait
# for text-only runs, so pass '832*480' explicitly. With a src_video the output follows the
# source's aspect ratio at the same area.
SIZE='832*480'

PROMPT=${PROMPT:-"Two tabby cats wearing red boxing gloves and boxing shorts spar in a boxing ring under bright spotlights, trading punches. A dalmatian watches from behind the ropes and a crowd sits in the dark background. Realistic, cinematic lighting, smooth natural motion."}

# run_vace CASE [extra vace_wan_inference.py args...]
run_vace() {
  local case=$1; shift
  local save_dir=$OUT_ROOT/$case/$MODEL; [[ ${VACE_DIT_BF16:-0} == 1 ]] && save_dir+=-bf16
  mkdir -p "$save_dir"
  cd "$VACE_DIR"
  echo "[vace smoke] case=$case model=$MODEL seed=$SEED steps=$STEPS CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
  echo "[vace smoke] VACE_DIT_BF16=${VACE_DIT_BF16:-0}"
  "$PY" "$SMOKE_DIR/vace_launch.py" \
    --model_name "vace-$MODEL" \
    --ckpt_dir "$CKPT_DIR" \
    --size "$SIZE" \
    --frame_num 81 \
    --sample_steps "$STEPS" \
    --base_seed "$SEED" \
    "${MEM_FLAGS[@]}" \
    --prompt "$PROMPT" \
    --save_dir "$save_dir" \
    "$@" 2>&1 | tee "$save_dir/run.log"
}
