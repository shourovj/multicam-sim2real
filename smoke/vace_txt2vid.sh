#!/usr/bin/env bash
# VACE smoke case 1: text-only (no src_video / src_mask). 81 frames, 832x480.
#   MODEL=14B|1.3B bash smoke/vace_txt2vid.sh
source "$(dirname "${BASH_SOURCE[0]}")/vace_common.sh"
run_vace txt2vid
