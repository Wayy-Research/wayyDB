# WayyDB API Docker Image
# Supports: Fly.io, Render, HuggingFace Spaces

FROM python:3.12

# Install only essential build tools via apt (g++ for C++ compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user (required by HF Spaces, good practice everywhere)
RUN useradd -m -u 1000 user
RUN mkdir -p /home/user/data/wayydb /data/wayydb && \
    chown -R user:user /home/user /data

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    WAYY_DATA_PATH=/data/wayydb \
    PORT=8080

WORKDIR $HOME/app

# Install build tools via pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir cmake ninja scikit-build-core pybind11 numpy build

# Copy source
COPY --chown=user . .

# Build and install wayyDB
RUN pip install --no-cache-dir -v . && \
    pip install --no-cache-dir -r api/requirements.txt

# Support both Fly.io (8080) and HuggingFace (7860)
EXPOSE 8080 7860

# Use shell form to allow $PORT substitution
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}
