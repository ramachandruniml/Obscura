FROM python:3.11-slim AS base

# ffmpeg for video decode/encode/mux; libgl + libglib for opencv-headless runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /bin/uv

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# torch variant: "cpu" (default, small, CI-friendly) or e.g. "cu121" for CUDA wheels.
ARG BUILD_TORCH_VARIANT=cpu

COPY pyproject.toml ./
# Install torch first from the dedicated index so the right variant is pinned,
# then the rest of the deps from PyPI.
RUN uv pip install --index-url "https://download.pytorch.org/whl/${BUILD_TORCH_VARIANT}" \
        "torch>=2.5,<2.7" "torchvision>=0.20,<0.22" \
    && uv pip install ".[dev]"

COPY . .
RUN uv pip install --no-deps -e .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
