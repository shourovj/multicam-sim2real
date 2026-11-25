#!/usr/bin/env bash
# Serve the spec writer's model (and the optional render judge) with vLLM on one GPU, OpenAI-compatible API.
#   CUDA_VISIBLE_DEVICES=6 sim/physx/spec/serve_llm.sh            # port 8016
# Model: Qwen3.8-27B, the strongest open vision-language model that fits one 80 GB A100 (Apache-2.0), in Qwen's own
# FP8 checkpoint: on an A100 (weight-only FP8) it decodes 83 tok/s with MTP against 53 for BF16 and 27 for BF16 without
# MTP, which decides how many SceneAgent iterations fit a 15-minute budget. P2S_LLM_MODEL=Qwen/Qwen3.8-27B for BF16.
# Weights and the vLLM env live in the gitignored sim/physx/.deps (see sim/physx/agent/README.md).
set -eu
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
deps=$here/.deps
model=${P2S_LLM_MODEL:-Qwen/Qwen3.8-27B-FP8}
port=${P2S_LLM_PORT:-8016}
source "$deps/envs/vlm/.venv/bin/activate"
export HF_HUB_CACHE=$deps/hf HF_HUB_OFFLINE=1
# MTP: the model's own multi-token-prediction head drafts tokens that the full model verifies (same output, faster).
exec vllm serve "$model" --port "$port" --served-model-name p2s-llm \
  --max-model-len 65536 --max-num-seqs 8 --gpu-memory-utilization 0.92 --limit-mm-per-prompt '{"image": 4}' \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --reasoning-parser qwen3 "$@"
