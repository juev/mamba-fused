# Fused Mamba CUDA kernels baked into the image — drop-in for vast.ai RTX 4090 runs.
#
# Replaces the ~15 min per-run source compile (build_kernels.sh in juev/llm) with a
# one-time CI build. The base devel image ships nvcc, so causal-conv1d and mamba-ssm
# cross-compile for sm_89 (RTX 4090) WITHOUT a GPU at build time — TORCH_CUDA_ARCH_LIST
# is what makes that work. +PTX keeps forward-compat (newer GPUs JIT from PTX).
#
# Pins mirror the verified recipe (vast-ai skill, 2026-06-16): torch 2.3.0 / cu121,
# causal-conv1d v1.2.0.post2, mamba v1.2.0.post1, transformers 4.40.2 (has Mamba +
# the generation aliases mamba-ssm 1.2.0 still imports).
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    TORCH_CUDA_ARCH_LIST="8.9+PTX" \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    MAMBA_FORCE_BUILD=TRUE \
    MAX_JOBS=2

# The base image ships neither git (needed to clone the kernel sources) nor a host
# C++ toolchain for nvcc. On vast.ai these come from the instance provisioning, not
# the image — a clean docker build must install them explicitly.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git build-essential \
 && rm -rf /var/lib/apt/lists/*

# causal-conv1d first (mamba-ssm links against it). --no-build-isolation so it builds
# against the image's torch, not a fresh PyPI wheel (ABI mismatch otherwise).
RUN git clone -q --branch v1.2.0.post2 --depth 1 https://github.com/Dao-AILab/causal-conv1d.git /tmp/causal-conv1d \
 && pip install --no-build-isolation /tmp/causal-conv1d \
 && rm -rf /tmp/causal-conv1d

RUN git clone -q --branch v1.2.0.post1 --depth 1 https://github.com/state-spaces/mamba.git /tmp/mamba \
 && pip install --no-build-isolation /tmp/mamba \
 && rm -rf /tmp/mamba

RUN pip install --no-cache-dir "transformers==4.40.2" datasets tqdm

# Fail the build here if kernels don't import — better than discovering it on vast.
RUN python -c "import causal_conv1d, mamba_ssm; print('kernels import OK', causal_conv1d.__version__, mamba_ssm.__version__)"
