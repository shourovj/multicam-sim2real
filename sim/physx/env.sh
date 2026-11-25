# Paths for the repo's scripts and for sim/physx. Source it in every new shell: source sim/physx/env.sh
# Third-party tools (Python env, Blender, PhysX build, CUDA 13, robot model) live in sim/physx/.deps (gitignored).
# Outputs and caches go to /nas/raghav/sim2real, an SMB share: plain files only, no symlinks or venvs there.
P2S_HOME=$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)
export P2S_HOME
export S2R_ROOT=$P2S_HOME/.deps
export HF_HUB_CACHE=/mnt/Data/sj37434/hf/hub              # shared, read-only
export COSMOS_HF_CACHE=$HF_HUB_CACHE
export HF_HUB_OFFLINE=1
export S2R_DATA=/nas/raghav/sim2real/sim2real-data
export S2R_CACHE=/nas/raghav/sim2real/.cache/sim2real
export UV_CACHE_DIR=$S2R_ROOT/uv-cache                    # uv hard-links from its cache: keep it off the SMB share
export CUDA_CACHE_MAXSIZE=4294967296
