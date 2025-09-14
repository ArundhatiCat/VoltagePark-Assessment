# -------- GPU (H100) build with CUDA & PyTorch --------
# Choose a recent PyTorch image with CUDA 12.4+ runtime (works with 12.7 drivers on host)
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git-lfs \
 && rm -rf /var/lib/apt/lists/*

# Enable large file pulls from Hugging Face if needed
RUN git lfs install

# Create app dir
WORKDIR /app

# Copy only dependency file first for layer caching
COPY requirements.txt /app/requirements.txt

# Python deps (GPU)
# diffusers/transformers/accelerate/huggingface_hub/psutil + uvicorn/fastapi
# xFormers is optional; skip if you hit compatibility issues.
RUN pip install --no-cache-dir -r requirements.txt \
    || true

# If you don’t use requirements.txt, you can do:
# RUN pip install --no-cache-dir fastapi uvicorn[standard] diffusers transformers accelerate huggingface_hub psutil

# Copy your source last
COPY . /app

# Hugging Face cache (to persist models between container restarts when mounted)
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p ${HF_HOME}

# (Optional) Torch CUDA settings for H100
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV PYTHONUNBUFFERED=1

# Expose FastAPI port
EXPOSE 8000

# Healthcheck (optional)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1',8000)); print('ok')" || exit 1

# Entrypoint
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
