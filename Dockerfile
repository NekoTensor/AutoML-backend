# HF Spaces (Docker SDK) builds and runs this directly.
# Spaces route external traffic to port 7860 by default.

FROM python:3.11-slim

WORKDIR /app

# System deps some wheels (torch, onnx) may need at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# These directories are writable scratch space at runtime — HF Spaces'
# filesystem is ephemeral (wiped on restart/sleep), which is fine for
# this app since the workflow is upload -> process -> download in one
# sitting, not long-term storage.
RUN mkdir -p uploads models reports

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
