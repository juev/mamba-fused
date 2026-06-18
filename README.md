# mamba-fused

Docker image with prebuilt fused Mamba CUDA kernels (`causal-conv1d` + `mamba-ssm`)
for vast.ai RTX 4090 runs. Replaces the ~15 min per-run source compile with a
one-time CI build, cutting instance setup from ~17 min to a plain image pull.

## Image

```
ghcr.io/juev/mamba-fused:cu121-torch23-sm89
```

Base `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel`. Kernels cross-compiled for
**sm_89** (RTX 4090) via `TORCH_CUDA_ARCH_LIST=8.9+PTX` — no GPU needed at build.
Pins: torch 2.3.0 / cu121, causal-conv1d v1.2.0.post2, mamba v1.2.0.post1,
transformers 4.40.2 (verified recipe, see `juev/llm` vast-ai skill).

## Use on vast.ai

```bash
vastai create instance <offer_id> \
  --image ghcr.io/juev/mamba-fused:cu121-torch23-sm89 \
  --disk 40 --ssh --direct
# then, on the box, straight to the gate — no build step:
python check_cuda_kernels.py    # must print FAST PATH active
```

## Build

CI builds on every change to `Dockerfile` or the workflow (`workflow_dispatch` for
manual runs). Native amd64 GitHub runner, pushes via the built-in `GITHUB_TOKEN`.

**One-time after the first successful run:** set the package public
(repo → Packages → mamba-fused → Package settings → Change visibility → Public),
so vast pulls anonymously. Otherwise pass a `read:packages` token via `--login`.

## Rebuild triggers

Bump a pin or the CUDA arch only by editing the `Dockerfile`; the tag stays stable,
so re-running the workflow overwrites the image in place. For a different GPU arch,
add its `sm_XX` to `TORCH_CUDA_ARCH_LIST` and a matching tag.
