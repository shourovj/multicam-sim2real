#!/usr/bin/env bash
# VACE smoke case 2: first-frame extension (VACE 'frameref' task, mode firstframe).
# src_video = frame 0 + 80 gray frames, src_mask = black on frame 0, white after.
# Inputs come from smoke/vace_prep_inputs.sh.
#   MODEL=14B|1.3B bash smoke/vace_firstframe.sh
source "$(dirname "${BASH_SOURCE[0]}")/vace_common.sh"
IN=$OUT_ROOT/firstframe/inputs
for f in "$IN/src_video-frameref.mp4" "$IN/src_mask-frameref.mp4"; do
  [[ -f $f ]] || { echo "missing $f; run smoke/vace_prep_inputs.sh first" >&2; exit 1; }
done
run_vace firstframe --src_video "$IN/src_video-frameref.mp4" --src_mask "$IN/src_mask-frameref.mp4"
