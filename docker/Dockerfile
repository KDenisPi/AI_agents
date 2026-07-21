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

COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
