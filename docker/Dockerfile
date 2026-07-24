# Build from the project root so the app source is in the build context:
#   docker build -f docker/Dockerfile -t ai-agents .
#
# No default CMD - pick the entrypoint at `docker run` time, e.g.:
#   docker run --rm -p 8000:8000 ai-agents python McpMultiHttpServer.py
#   docker run --rm --env-file .env ai-agents python Collector.py

FROM python:3.12-slim

WORKDIR /app

# libportaudio2 is the runtime shared library sounddevice loads via ctypes.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# torch backs only the SNAC vocoder in text_to_voice.py, which is small and
# fast enough on CPU - the language models run on a separate Ollama host, not
# in here. Taking it from PyTorch's CPU index keeps ~2.5 GB of unused CUDA
# libraries out of the image. --index-url rather than --extra-index-url so
# only that index is consulted: with both in play pip compares versions
# across them and would pull the CUDA wheel whenever PyPI happens to be
# ahead. Kept above the requirements copy so editing that file doesn't
# invalidate this layer.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
