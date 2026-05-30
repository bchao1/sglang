# Local Environment & Patch Changelog

## Context
Machine: GPU server with NVIDIA RTX A6000 (SM86, Ampere), driver 560.35.05 (CUDA 12.6 max).
SGLang was configured for Hopper (SM90) / CUDA 13.0 out of the box. These changes make it work on Ampere + CUDA 12.6.

---

## Python environment (`genAI` conda env)

### torch — reinstalled for CUDA 12.6
```
pip install --force-reinstall torch==2.11.0+cu126 torchvision \
    --index-url https://download.pytorch.org/whl/cu126
```
Was: `torch 2.11.0+cu130` (requires driver ≥ 570, this machine has 560).  
Now: `torch 2.11.0+cu126` (matches driver 560.35.05 / CUDA 12.6).

### sgl-kernel — rebuilt from source
```
cd sgl-kernel
PATH=/usr/local/cuda/bin:$PATH \
CMAKE_ARGS="-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
            -DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)') \
            -DSGL_KERNEL_ENABLE_FA3=OFF \
            -DSGL_KERNEL_ENABLE_FLASHMLA=OFF" \
pip install . --no-build-isolation
```
The PyPI wheel (`0.4.3`) was compiled for SM90/SM100 + CUDA 13.0. Building from source produces SM86-compatible binaries linked against CUDA 12.6.

**Flags:**
- `-DSGL_KERNEL_ENABLE_FA3=OFF` — skip `flash_ops` build (SM90 Hopper-specific; SM90 `.cu` instantiations crash nvcc 12.6 with SIGSEGV)
- `-DSGL_KERNEL_ENABLE_FLASHMLA=OFF` — skip `flashmla_ops` build (SM90/SM100 only; uses `__nv_fp8_e8m0` which requires CUDA 13.0)

### CUDA 13.x pip packages removed
Packages like `nvidia-cuda-runtime 13.0.x`, `nvidia-nccl-cu13`, etc. were installed as dependencies of the old cu130 wheel. They were removed because the GPU worker subprocess would load CUDA 13.0 runtime (`libcudart.so.13`), which the driver can't support. Keep only `nvidia-*-cu12` packages.

---

## Code changes (branch: `fix/cuda126-sgl-kernel-build`)

All changes are in `sgl-kernel/`. These are correctness fixes suitable for upstream PR — not Ampere-specific hacks.

### 1. `sgl-kernel/CMakeLists.txt`

**Fix: CUDA_VERSION not populated by modern CMake**  
`cmake`'s new CUDA language support sets `CMAKE_CUDA_COMPILER_VERSION`, not the legacy `CUDA_VERSION` variable. All version guards were comparing against an empty string.  
```cmake
if (NOT CUDA_VERSION AND CMAKE_CUDA_COMPILER_VERSION)
    string(REGEX MATCH "^[0-9]+\\.[0-9]+" CUDA_VERSION "${CMAKE_CUDA_COMPILER_VERSION}")
endif()
```

**Fix: SM100 MXFP8 sources guarded behind CUDA ≥ 13.0**  
`es_sm100_mxfp8_blockscaled*.cu` uses `__nv_fp8_e8m0` introduced in CUDA 13.0. Previously unconditional.  
```cmake
if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "13.0" OR SGL_KERNEL_ENABLE_SM100A)
    list(APPEND SOURCES "csrc/expert_specialization/es_sm100_mxfp8_blockscaled.cu" ...)
    add_compile_definitions(SGL_KERNEL_HAVE_SM100_MXFP8=1)
endif()
```

**Fix: `SGL_KERNEL_ENABLE_FA3=OFF` user flag respected**  
The cmake auto-enable for CUDA ≥ 12.4 used `set()` which overwrote the user's `-DSGL_KERNEL_ENABLE_FA3=OFF` cache entry.  
```cmake
if (NOT DEFINED CACHE{SGL_KERNEL_ENABLE_FA3})
    set(SGL_KERNEL_ENABLE_FA3 ON)
endif()
```

**Add: `SGL_KERNEL_ENABLE_FLASHMLA` option**  
FlashMLA is SM90-only. Added an opt-out for pre-SM90 builds.  
```cmake
option(SGL_KERNEL_ENABLE_FLASHMLA "Build FlashMLA (SM90/SM100 only)" ON)
if (SGL_KERNEL_ENABLE_FLASHMLA)
    include(cmake/flashmla.cmake)
endif()
```

### 2. `sgl-kernel/include/sgl_kernel_ops.h` + `csrc/common_extension.cc`

**Fix: SM100 MXFP8 op declarations/registrations gated on `SGL_KERNEL_HAVE_SM100_MXFP8`**  
Without the source files compiled in, the linker fails on undefined `es_sm100_mxfp8_blockscaled_grouped_*` symbols. Wrapped with `#if defined(SGL_KERNEL_HAVE_SM100_MXFP8)`.

### 3. `sgl-kernel/python/sgl_kernel/flash_attn.py`

**Fix: `flash_ops` import made lazy (deferred to call time)**  
The old code raised `ImportError` at module load time when `flash_ops.abi3.so` was absent:
```python
# Before — crashes at import, blocks all sglang diffusion startup
from sgl_kernel import flash_ops  # raises if .so missing
```
Fixed: import is attempted silently; functions raise only when actually called.

---

## Diffusion generation

### Attention backend
FLUX.1-dev and Z-Image default to FA (FlashAttention 3) which requires `flash_ops.abi3.so`. On Ampere without that `.so`, use:
```
--attention-backend torch_sdpa
```
This selects PyTorch's built-in Scaled Dot Product Attention. Quality is identical; it's slightly slower than FA3 but fully correct on SM86.

### Z-Image status
Z-Image requires CUTLASS DSL (`nvidia-cutlass-dsl-libs-cu13`) which pulls in `cuda-python 13.3.1` → CUDA 13 runtime → driver error. **Z-Image cannot run on this machine** until the driver is updated to support CUDA 13 (requires driver ≥ 570.x).

FLUX.1-dev works fully. ✓

---

## Test script
`scratch/test_diffusion_gen.sh` — FLUX.1-dev at 50 steps, TORCH_SDPA backend, GPU auto-selected via `select_gpu.sh`.  
Z-Image is commented out pending driver update.

---

## Upstream PR notes
The `fix/cuda126-sgl-kernel-build` branch contains all 4 modified files. The fixes are generic (not Ampere-specific) and should be submitted upstream:
- PRs can be split: (1) CMakeLists CUDA_VERSION + SM100 guard, (2) FA3/FlashMLA override correctness, (3) lazy flash_attn import
- The `-FA3=OFF -FLASHMLA=OFF` build flags are local build choices, not code patches — no need to land in upstream
