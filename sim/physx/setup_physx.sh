#!/bin/bash
# Fetch a CUDA 13 toolkit (pip wheels, no system install) and build PhysX 5.9.1 with the p2s patch:
#   sim/physx/setup_physx.sh [deps_dir]          (default: sim/physx/.deps; takes ~20-40 min on many cores)
# The patch: CUDA 13 build fixes, sm_80 (A100) GPU code only, no snippets, and FEM plasticity re-enabled
# (PxDeformableVolumeMaterial::setDeformThreshold / setDeformLowLimitRatio / setDeformHighLimitRatio).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DEPS=$(realpath -m "${1:-$HERE/.deps}")
mkdir -p "$DEPS"

if [ ! -x "$DEPS/cuda13/bin/python" ]; then
  uv venv -q --python 3.11 "$DEPS/cuda13"
  uv pip install -q --python "$DEPS/cuda13/bin/python" nvidia-cuda-nvcc==13.0.88 nvidia-cuda-runtime==13.0.96 \
    nvidia-cuda-crt==13.0.88 nvidia-cuda-cccl==13.0.85 nvidia-nvvm==13.0.88
fi
CUDA=$DEPS/cuda13/lib/python3.11/site-packages/nvidia/cu13
[ -e "$CUDA/lib/libcudart.so" ] || ln -s libcudart.so.13 "$CUDA/lib/libcudart.so"

if [ ! -d "$DEPS/physx-src" ]; then
  git clone -q https://github.com/NVIDIA-Omniverse/PhysX.git "$DEPS/physx-src"
  git -C "$DEPS/physx-src" checkout -q 4f5f477   # "Update to Omniverse Physics 110.3 and PhysX SDK 5.9.1"
  git -C "$DEPS/physx-src" apply "$HERE/third_party/physx-5.9.1-s2r.patch"
fi

cd "$DEPS/physx-src/physx"
export PATH=$CUDA/bin:$PATH CUDACXX=$CUDA/bin/nvcc CUDA_PATH=$CUDA
./generate_projects.sh linux-gcc
cmake -S compiler/linux-gcc-release -B compiler/linux-gcc-release \
  -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA" > /dev/null
cmake --build compiler/linux-gcc-release -j "$(nproc)"
ls bin/linux.x86_64/release/libPhysXGpu_64.so bin/linux.x86_64/release/libPhysX_static_64.a
echo "PhysX 5.9.1 (p2s patch) built in $DEPS/physx-src/physx/bin/linux.x86_64/release"
