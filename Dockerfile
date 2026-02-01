# WayyDB API Docker Image for Hugging Face Spaces
FROM python:3.12-slim

# Install build dependencies (including Python headers for pybind11)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user

# Create data directory before switching user
RUN mkdir -p /home/user/data/wayydb && chown -R user:user /home/user

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    WAYY_DATA_PATH=/home/user/data/wayydb \
    PORT=7860

WORKDIR $HOME/app

# Copy source
COPY --chown=user . .

# Build and install wayyDB (verbose for debugging)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir build scikit-build-core pybind11 numpy cmake ninja && \
    pip install --no-cache-dir -v . && \
    pip install --no-cache-dir -r api/requirements.txt

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
