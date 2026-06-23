# Fused Mamba kernels baked into the image — drop-in for vast.ai RTX 4090 (sm_89).
#
# Packages are IDENTICAL to the validated juev/llm stack (2026-06-18, transformers 5):
# torch 2.10.0+cu126, causal-conv1d 1.6.2.post1, mamba-ssm 2.3.2.post1, transformers 5,
# einops — installed from the SAME prebuilt wheels, so there is no source compile and
# no nvcc. That also lets us use the smaller *runtime* base instead of *devel*.
#
# The HF `kernels` library is deliberately NOT installed: when present, transformers
# prefers a Hub kernel whose fully-fused mamba_inner_fn can't find causal_conv1d_cuda,
# and Mamba training breaks. The classic prebuilt wheels expose it correctly.
#
# Prebuilt wheels exist up to torch 2.10 (not 2.12 yet) → torch 2.10 base image.
# cu12torch2.10 matches this image's torch; cp312 matches its CPython.
FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

# This base image marks its python as externally-managed (PEP 668); pip refuses to
# install without an override. One env var covers every pip call below.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# vast.ai injects the SSH key into /root/.ssh/authorized_keys at boot; sshd StrictModes
# then refuses login ("bad ownership or modes") if /root, /root/.ssh, or the
# authorized_keys file are group/world-writable. The runtime base leaves them too open.
#
# Fix directories + pre-create authorized_keys with 600 (vast may append, not overwrite).
# Also nuke StrictModes in sshd_config as a backstop — the instance is single-user root,
# temporary, and key-only auth; the check adds no value here.
RUN mkdir -p /root/.ssh \
 && touch /root/.ssh/authorized_keys \
 && chmod 700 /root /root/.ssh \
 && chmod 600 /root/.ssh/authorized_keys \
 && if [ -f /etc/ssh/sshd_config ]; then \
      sed -i 's/^#*StrictModes.*/StrictModes no/' /etc/ssh/sshd_config; \
    fi

ARG CAUSAL_CONV1D_WHL=https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
ARG MAMBA_SSM_WHL=https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# --no-deps: the wheels declare a loose `torch` requirement; without it pip would pull
# a newer torch (cu130) over the image's matched 2.10/cu126 and break the kernel ABI.
RUN pip install --no-cache-dir --no-deps "$CAUSAL_CONV1D_WHL" "$MAMBA_SSM_WHL" \
 && pip install --no-cache-dir einops "transformers>=5,<6" datasets tqdm

# Fail the build here if the kernels don't import (ABI mismatch surfaces on load) —
# better than discovering it on vast. Runs without a GPU on the CI runner.
RUN python -c "import causal_conv1d, mamba_ssm; print('kernels import OK', causal_conv1d.__version__, mamba_ssm.__version__)"
