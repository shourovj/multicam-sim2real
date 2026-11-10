#!/usr/bin/env bash
# Wan-Move smoke test: runs the repo's shipped example (examples/example.jpg +
# example_tracks.npy + example_visibility.npy + the README prompt) on ONE GPU.
# Uses cuda:0 of whatever CUDA_VISIBLE_DEVICES the caller sets; does not touch it.
#   bash smoke/wanmove_example.sh
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
ROOT="$S2R_ROOT"
WANMOVE="$ROOT/Wan-Move"
OUT_DIR="$S2R_DATA/smoke/wanmove"
SEED="${SEED:-42}"
mkdir -p "$OUT_DIR"

cd "$WANMOVE"  # generate.py resolves examples/ and the default ckpt_dir relative to here
# Wan-Move-14B-480P is a symlink to the HF cache snapshot (63a21fd...).
.venv/bin/python generate.py \
  --task wan-move-i2v \
  --size 480*832 \
  --ckpt_dir ./Wan-Move-14B-480P \
  --image examples/example.jpg \
  --track examples/example_tracks.npy \
  --track_visibility examples/example_visibility.npy \
  --prompt "A laptop is placed on a wooden table. The silver laptop is connected to a small grey external hard drive and transfers data through a white USB-C cable. The video is shot with a downward close-up lens." \
  --frame_num 81 \
  --base_seed "$SEED" \
  --offload_model True \
  --t5_cpu \
  --dtype bf16 \
  --save_file "$OUT_DIR/example_seed${SEED}.mp4" \
  2>&1 | tee "$OUT_DIR/example_seed${SEED}.log"
