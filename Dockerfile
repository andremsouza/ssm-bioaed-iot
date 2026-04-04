# === BioAED: Bioacoustic Audio Event Detection ===
# Base image: PyTorch 2.10 + CUDA 12.6 + cuDNN 9 (verified mamba-ssm compatible)
FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-devel

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv (ultra-fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /workspace

# Copy dependency files first (for Docker layer caching)
COPY pyproject.toml uv.lock* ./

# Sync dependencies with CUDA extras
RUN uv sync --extra cuda --no-dev --frozen || uv sync --extra cuda --no-dev

# Copy source code
COPY . .

# Install the project in editable mode
RUN uv sync --extra cuda --frozen || uv sync --extra cuda

# Default entrypoint
ENTRYPOINT ["uv", "run", "python", "-m"]
CMD ["bioaed.train", "--help"]
