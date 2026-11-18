# syntax=docker/dockerfile:1
# Multi-stage Dockerfile for Edge-Deployable Space Debris Detection System
# Designed for multi-architecture deployment across ARM64 (Raspberry Pi 4/5, Jetson) and x86_64.

# ==============================================================================
# Stage 1: Build & Dependencies
# ==============================================================================
FROM python:3.10-slim AS builder

WORKDIR /build

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install edge runtime dependencies
COPY requirements-edge.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-edge.txt && \
    pip install --no-cache-dir skyfield pytest

# ==============================================================================
# Stage 2: Minimal Edge Runtime (Default target)
# ==============================================================================
FROM python:3.10-slim AS edge-runtime

LABEL maintainer="Edge Vision & Machine Learning Team" \
      description="Edge-ready space debris and satellite detection container" \
      version="1.0"

# Install minimal runtime shared libraries (OpenCV headless & OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create unprivileged non-root service account
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Copy application source code, configurations, and cached TLE catalog
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY edge/ /app/edge/
COPY tests/ /app/tests/
COPY pytest.ini /app/
COPY gp.php /app/

# Set ownership to unprivileged user
RUN chown -R appuser:appuser /app
USER appuser

# Health check verifying edge runtime integrity
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import onnxruntime, cv2, numpy, skyfield; print('Edge runtime healthy')" || exit 1

# Default entrypoint for CLI inference
ENTRYPOINT ["python", "src/infer.py"]
CMD ["--help"]
