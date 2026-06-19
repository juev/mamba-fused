# mamba-fused

Docker image with prebuilt fused Mamba CUDA kernels (`causal-conv1d` + `mamba-ssm`)
for vast.ai RTX 4090 runs. Bakes the validated `juev/llm` kernel stack into an image,
cutting instance setup from a ~15 min source compile (or a per-run `pip install`) to a
plain image pull.

## Image

```
ghcr.io/juev/mamba-fused:cu126-torch210-sm89
```

Base `pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime`. The kernels are installed from
the **same prebuilt wheels** as the validated `juev/llm` stack — so the image's packages
are identical to a local `build_kernels.sh` install. No source compile, no nvcc.

Packages (transformers 5 stack, validated 2026-06-18 on RTX 4090 — FAST PATH, ~441K tok/s):

| package | version |
|---|---|
| torch | 2.10.0+cu126 |
| causal-conv1d | 1.6.2.post1 (`cu12torch2.10` wheel) |
| mamba-ssm | 2.3.2.post1 (`cu12torch2.10` wheel) |
| transformers | >=5,<6 |
| einops | latest |

The HF `kernels` library is deliberately **absent**: when present, transformers prefers a
Hub kernel whose fully-fused `mamba_inner_fn` can't find `causal_conv1d_cuda`, breaking
Mamba training. The classic prebuilt wheels expose it correctly.

## Use on vast.ai

```bash
vastai create instance <offer_id> \
  --image ghcr.io/juev/mamba-fused:cu126-torch210-sm89 \
  --disk 40 --ssh --direct
# then, on the box, straight to the gate — no build, no pip install:
python check_cuda_kernels.py    # must print FAST PATH active
```

### Previous image (transformers 4.40)

`ghcr.io/juev/mamba-fused:cu121-torch23-sm89` — torch 2.3 / cu121, transformers 4.40.2,
source-compiled causal-conv1d 1.2.0 / mamba 1.2.0. Kept for runs that need transformers
4.x; otherwise use the tag above.

## Build

CI builds on every change to `Dockerfile` or the workflow (`workflow_dispatch` for manual
runs). Native amd64 GitHub runner, pushes via the built-in `GITHUB_TOKEN`.

**One-time after the first successful run:** set the package public
(repo → Packages → mamba-fused → Package settings → Change visibility → Public),
so vast pulls anonymously. Otherwise pass a `read:packages` token via `--login`.

## Rebuild triggers

Bump a pin by editing the `Dockerfile` (the wheel URLs / version specs). New stacks get a
new tag (e.g. `cu126-torch210-sm89`) so older images stay pullable at their tags. For a
different GPU arch, pick matching wheels and add a `sm_XX` tag.
