# HF Spaces (Docker SDK) builds and runs this directly.
# Spaces route external traffic to port 7860 by default.

FROM python:3.11-slim

WORKDIR /app

# System deps some wheels (torch, onnx) may need at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install torch from PyTorch's CPU index BEFORE the rest of requirements.
#
# PyPI's torch is the CUDA build: it drags in ~1.9 GB of nvidia-* wheels
# plus triton (209 MB), none of which can be used here — Spaces of this
# tier have no GPU, and the pipeline runs on CPU. That payload is also
# what broke the build: the triton download truncated partway through and
# pip failed the whole install on a hash mismatch. Downloading 2.8 GB that
# gets thrown away is a coin flip on every rebuild.
#
# The CPU wheel declares no nvidia-* or triton dependencies at all, so this
# removes the failure mode rather than retrying it, and cuts ~2.8 GB from
# the build (torch alone goes 906 MB -> 167 MB).
#
# requirements.txt still pins `torch==2.5.1`; that specifier has no local
# version label, so per PEP 440 it is satisfied by the `2.5.1+cpu` build
# installed here and the next step will not pull the CUDA one back in.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# These directories are writable scratch space at runtime — HF Spaces'
# filesystem is ephemeral (wiped on restart/sleep), which is fine for
# this app since the workflow is upload -> process -> download in one
# sitting, not long-term storage.
RUN mkdir -p uploads models reports

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
