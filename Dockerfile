# WayyDB API Docker Image for Hugging Face Spaces
FROM python:3.12-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy source
COPY --chown=user . .

# Build and install wayyDB
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir build scikit-build-core pybind11 numpy && \
    pip install --no-cache-dir . && \
    pip install --no-cache-dir -r api/requirements.txt

# Create data directory
RUN mkdir -p $HOME/data/wayydb

ENV WAYY_DATA_PATH=$HOME/data/wayydb
ENV PORT=7860

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
