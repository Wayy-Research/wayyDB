# WayyDB API Docker Image for Hugging Face Spaces
FROM python:3.12

# Install only essential build tools via apt (g++ for C++ compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
RUN mkdir -p /home/user/data/wayydb && chown -R user:user /home/user

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    WAYY_DATA_PATH=/home/user/data/wayydb \
    PORT=7860

WORKDIR $HOME/app

# Install build tools via pip (more reliable than apt on HF)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir cmake ninja scikit-build-core pybind11 numpy build

# Copy source
COPY --chown=user . .

# Build and install wayyDB
RUN pip install --no-cache-dir -v . && \
    pip install --no-cache-dir -r api/requirements.txt

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
