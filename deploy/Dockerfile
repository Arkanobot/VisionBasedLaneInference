# Build recipe executed by the host (Hugging Face Spaces). Nothing here runs on
# the development machine.
FROM python:3.11-slim

# opencv-python-headless still needs libGL's stub-free deps for some codecs, and
# ffmpeg is required to write browser-playable H.264 for the video tab.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs as uid 1000; the app writes rendered videos under /app/outputs.
RUN useradd -m -u 1000 app
WORKDIR /app

# Install deps first so the layer caches across code pushes. The CPU wheel index
# matters: the default torch wheel carries CUDA and is ~4x larger for no benefit
# on a CPU host.
COPY --chown=app:app requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.13.0 torchvision==0.28.0 \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app . .

USER app
ENV PYTHONPATH=/app/lane_inference \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    HF_HOME=/tmp/hf

# Spaces routes to 7860 by default.
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "server.api:app", "--host", "0.0.0.0", "--port", "7860"]
