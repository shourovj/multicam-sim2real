#!/usr/bin/env bash
# Build the preprocessed inputs for the firstframe and ffdepth VACE smoke cases.
# Runs on CPU (CUDA_VISIBLE_DEVICES= bash legacy/smoke/vace_prep_inputs.sh, ~a few minutes) or on the
# GPU given by the environment. Outputs under $S2R_DATA/smoke/vace/.
source "$(dirname "${BASH_SOURCE[0]}")/vace_common.sh"
cd "$VACE_DIR"
"$PY" "$SMOKE_DIR/vace_prep_inputs.py" "$@"
