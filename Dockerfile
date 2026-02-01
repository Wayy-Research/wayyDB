# WayyDB API Docker Image
FROM python:3.12-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source
COPY . .

# Build and install wayyDB
RUN pip install --upgrade pip && \
    pip install build scikit-build-core pybind11 numpy && \
    pip install . && \
    pip install -r api/requirements.txt

# Create data directory
RUN mkdir -p /data/wayydb

ENV WAYY_DATA_PATH=/data/wayydb
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
