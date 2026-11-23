#!/usr/bin/env bash
# One multicam run on Cosmos3-Nano-Sim-Transfer: build its inputs (gen/multicam_inputs.py), then generate with NVIDIA's
# Stage 3 recipe (.agents/skills/cosmos3-nano-sim-transfer/inference/SKILL.md): its defaults file, 4 fixed steps,
# text and control guidance 1, autoregressive with full history (no cache window, no attention sink), eager mode.
# Guardrails are off, as in every other run here (the guardrail model is gated).
#   EXP=configs/experiments/multicam_v1_2.yaml SCENE=intersection_4way VIEW=cctv COND=D SEED=0 bash gen/run_cosmos_simtransfer_multicam.sh
# Model, tokenizer and VAE are pinned (the revisions below match the experiment file); HF_HUB_OFFLINE keeps them from moving.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXP=${EXP:?}; SCENE=${SCENE:?}; VIEW=${VIEW:?}; COND=${COND:?}; SEED=${SEED:-0}
source "$REPO/env.sh"
HUB=${COSMOS_HF_CACHE:-/mnt/Data/sj37434/hf/hub}
export HF_HUB_CACHE=$HUB HF_HUB_OFFLINE=1 UV_CACHE_DIR=$S2R_CACHE/uv XDG_CACHE_HOME=$S2R_CACHE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled COSMOS_INTERNAL=0 COSMOS_TRAINING=1
unset LD_LIBRARY_PATH PYTHONPATH COSMOS_DISTILLATION_IMPL
CKPT=$HUB/models--nvidia--Cosmos3-Nano-Sim-Transfer/snapshots/45510d5b8cefa338b7abc01eb702cf1e9cdf6742
TOK=$HUB/models--nvidia--Cosmos3-Nano/snapshots/e59a53c25979a090fa8706c9acc0c254a6e89b92/text_tokenizer
VAE=$HUB/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth
for f in "$CKPT/config.json" "$TOK/tokenizer.json" "$VAE"; do [[ -e $f ]] || { echo "missing $f"; exit 1; }; done
cd "$S2R_ROOT/cosmos-framework"
OUT=$(.venv/bin/python "$REPO/gen/multicam_inputs.py" "$REPO/$EXP" "$SCENE" "$VIEW" "$COND" "$SEED" | sed -n 's/.*-> //p')
echo "[cosmos3-nano-sim-transfer multicam] $SCENE $VIEW $COND seed $SEED -> $OUT (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>})"
.venv/bin/python -m cosmos_framework.scripts.inference -i "$OUT/request.json" -o "$OUT/raw" \
  --checkpoint-path "$CKPT" --config-file "$CKPT/config.json" --credential-path "" \
  --experiment-overrides model.config.tokenizer.bucket_name= model.config.tokenizer.object_store_credential_path_pretrained= \
    "model.config.tokenizer.vae_path=$VAE" model.config.vlm_config.tokenizer.config_variant=hf \
    "model.config.vlm_config.tokenizer.pretrained_model_name=$TOK" model.config.vlm_config.pretrained_weights.enabled=False \
    model.config.teacher_forcing_frames_per_chunk=1 model.config.kv_cache_dtype=null \
  --defaults-file .agents/skills/cosmos3-nano-sim-transfer/inference/assets/sample_args.json \
  --autoregressive=True --cp-size 1 --cfgp-size 1 --dp-shard-size 1 --dp-replicate-size 1 \
  --seed "$SEED" --num-outputs 1 --max-num-seqs 1 --num-steps 4 --guidance 1 --control-guidance 1 \
  --kv-cache-inference-size 0 --attention-sink-size 0 \
  --no-use-torch-compile --no-use-cuda-graphs --no-condition-stream-encode --no-diffusion-cache --prompt-upsampling=False --no-guardrails \
  > "$OUT/run.log" 2>&1
cp "$OUT/raw/out/vision.mp4" "$OUT/out_video.mp4"
echo "[cosmos3-nano-sim-transfer multicam] done: $OUT/out_video.mp4"
