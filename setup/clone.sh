#!/usr/bin/env bash
# Clone each third-party repo into $S2R_ROOT at the commit the PoC (and Round 2, for VideoX-Fun) used. Safe to re-run:
# an existing clone is only checked out to the pinned commit, never reset or cleaned.
#   bash setup/clone.sh
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO/env.sh"
mkdir -p "$S2R_ROOT"

# An optional 4th column lists folders for a sparse clone (comma-separated): only those are checked out.
while read -r dir url commit sparse; do
  [[ -z $dir || $dir == \#* ]] && continue
  dest=$S2R_ROOT/$dir
  if [[ -d $dest/.git ]]; then
    echo "== $dir exists; checking out $commit"
  elif [[ -n ${sparse:-} ]]; then
    echo "== cloning $dir (sparse: $sparse)"
    git clone -q --filter=blob:none --sparse "$url" "$dest"
    git -C "$dest" sparse-checkout set ${sparse//,/ }
  else
    echo "== cloning $dir"
    git clone "$url" "$dest"
  fi
  git -C "$dest" fetch -q origin
  git -C "$dest" checkout -q "$commit"
done <<'EOF'
# dir             url                                                   commit
VACE              https://github.com/ali-vilab/VACE.git                 48eb44f1c4be87cc65a98bff985a26976841e9f3
Wan-Move          https://github.com/ali-vilab/Wan-Move.git             80c58a7d2ad175fa82a4d57f79f2a1415317dcfa
co-tracker        https://github.com/facebookresearch/co-tracker.git    82e02e8029753ad4ef13cf06be7f4fc5facdda4d
sam2              https://github.com/facebookresearch/sam2.git          2b90b9f5ceec907a1c18123530e92e794ad901a4
cosmos-framework  https://github.com/NVIDIA/cosmos-framework.git        5950e7b4ea3677b99a2de279a7ed7058b5446e42
VideoX-Fun        https://github.com/aigc-apps/VideoX-Fun.git           e8c314fbfa494d0429bbef605c9f6112b79a7ead
LTX-2             https://github.com/Lightricks/LTX-2.git               daa6b16fc7f0f1628f2c9e36a303e3312b333cc6
mujoco_menagerie  https://github.com/google-deepmind/mujoco_menagerie.git  370c43965323c0944b433b202aac5058b5f907a9  franka_emika_panda
EOF
