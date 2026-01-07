# base image
FROM openexplorer/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8-py310

# Install necessary packages and Python libraries (pynvm; soundfile; wandb)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip setuptools wheel
RUN python3.10 -m pip install --no-cache-dir pynvml soundfile wandb
