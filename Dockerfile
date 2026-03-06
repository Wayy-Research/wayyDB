# WayyDB API Docker Image
FROM python:3.12

# Install C++ toolchain and cmake via apt (more reliable than pip cmake)
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
RUN mkdir -p /home/user/data/wayydb /data/wayydb && \
    chown -R user:user /home/user /data

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    WAYY_DATA_PATH=/data/wayydb \
    PORT=8080

WORKDIR $HOME/app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir scikit-build-core pybind11 numpy build

COPY --chown=user . .

RUN pip install --no-cache-dir -v . && \
    pip install --no-cache-dir -r api/requirements.txt

EXPOSE 8080

CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}
