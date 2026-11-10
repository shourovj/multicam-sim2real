#!/usr/bin/env bash
# Clone each third-party repo into $S2R_ROOT at the commit the PoC used. Safe to re-run:
# an existing clone is only checked out to the pinned commit, never reset or cleaned.
#   bash setup/clone.sh
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO/env.sh"
mkdir -p "$S2R_ROOT"

while read -r dir url commit; do
  [[ -z $dir || $dir == \#* ]] && continue
  dest=$S2R_ROOT/$dir
  if [[ -d $dest/.git ]]; then
    echo "== $dir exists; checking out $commit"
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
EOF
