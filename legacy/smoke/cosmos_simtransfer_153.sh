#!/bin/bash
# Cosmos3-Nano-Sim-Transfer, example2 depth profile (full history, no sink), stretched to 153 frames.
# Copy of cosmos-logs/run-simtransfer-depth.sh without its hard-coded GPU. Run as:
#   legacy/smoke/measure.py --out <out>/measure.json -- legacy/smoke/cosmos_simtransfer_153.sh [frames]
set -euo pipefail
FRAMES=${1:-153}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO/env.sh"
ROOT=$S2R_ROOT
CONTROL=${CONTROL:-$S2R_DATA/smoke/cosmos/example2_depth_153.mp4}
OUT=${OUT:-$S2R_DATA/smoke/cosmos/out-$FRAMES}
mkdir -p "$OUT"
cd $ROOT/cosmos-framework
source .venv/bin/activate
unset PYTHONPATH COSMOS_DISTILLATION_IMPL LD_LIBRARY_PATH
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1 WANDB_MODE=disabled OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SKILL=$PWD/.agents/skills/cosmos3-nano-sim-transfer
hfpath() { uvx hf@2.4.0 download "$@" -q 2>/dev/null | tail -1; }   # -q prints only the local path
CKPT="$(hfpath nvidia/Cosmos3-Nano-Sim-Transfer --revision main --exclude 'assets/*')"
TOK="$(hfpath nvidia/Cosmos3-Nano --revision e59a53c25979a090fa8706c9acc0c254a6e89b92 --include 'text_tokenizer/*')/text_tokenizer"
VAE="$(hfpath Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e)"
ls "$CKPT/config.json" "$TOK" "$VAE" >/dev/null

# Request: the saved example2 depth request (legacy/configs/cosmos/) with the frame window set to $FRAMES.
# Optional env: PROMPT (replaces example2's prompt), NAME (output name), SEED (default 2025).
SEED=${SEED:-2025}
PROMPT=${PROMPT:-} NAME=${NAME:-} SEED=$SEED python3 - "$REPO/configs/cosmos/simtransfer_depth_request.json" "$OUT/request.json" "$CONTROL" "$FRAMES" <<'PY'
import json, os, sys
src, dst, control, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
r = json.load(open(src))
r.update(num_frames=n, max_frames=n, num_video_frames_per_chunk=n, name=os.environ["NAME"] or f"example2_depth_{n}")
r["seed"] = int(os.environ["SEED"])
if os.environ["PROMPT"]:
    r["prompt"] = os.environ["PROMPT"]
r["depth"] = {"control_path": control}
json.dump(r, open(dst, "w"), indent=2)
PY

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES FRAMES=$FRAMES"
python -m cosmos_framework.scripts.inference \
  -i "$OUT/request.json" -o "$OUT" \
  --checkpoint-path "$CKPT" --config-file "$CKPT/config.json" --credential-path "" \
  --experiment-overrides \
    model.config.tokenizer.bucket_name= \
    model.config.tokenizer.object_store_credential_path_pretrained= \
    "model.config.tokenizer.vae_path=$VAE" \
    model.config.vlm_config.tokenizer.config_variant=hf \
    "model.config.vlm_config.tokenizer.pretrained_model_name=$TOK" \
    model.config.vlm_config.pretrained_weights.enabled=False \
    model.config.teacher_forcing_frames_per_chunk=1 \
    model.config.kv_cache_dtype=null \
  --defaults-file "$SKILL/inference/assets/sample_args.json" \
  --autoregressive=True \
  --cp-size 1 --cfgp-size 1 --dp-shard-size 1 --dp-replicate-size 1 \
  --seed "$SEED" --num-outputs 1 --max-num-seqs 1 \
  --num-steps 4 --guidance 1 --control-guidance 1 \
  --kv-cache-inference-size 0 --attention-sink-size 0 \
  --no-use-torch-compile --no-use-cuda-graphs --no-condition-stream-encode \
  --no-diffusion-cache --prompt-upsampling=False --no-guardrails
