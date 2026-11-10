#!/usr/bin/env bash
# VACE smoke case 3: first frame + depth control (composition of frameref extension + depth control).
# src_video = real frame 0, then DepthV2 depth of source frames 1..80; src_mask = black on frame 0,
# white after. Inputs come from smoke/vace_prep_inputs.sh (rebuilt here if missing, on whatever
# device the environment gives).
#   MODEL=14B|1.3B bash smoke/vace_ffdepth.sh
source "$(dirname "${BASH_SOURCE[0]}")/vace_common.sh"
IN=$OUT_ROOT/ffdepth/inputs
if [[ ! -f $IN/src_video.mp4 || ! -f $IN/src_mask.mp4 ]]; then
  (cd "$VACE_DIR" && "$PY" "$SMOKE_DIR/vace_prep_inputs.py")
fi
run_vace ffdepth --src_video "$IN/src_video.mp4" --src_mask "$IN/src_mask.mp4"
