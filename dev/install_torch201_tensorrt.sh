#!/usr/bin/env bash

# This shell is used to install the env of torch-tensorrt.
# It should be noted that cuda-11.8 is required for torch/torch-tensorrt.

# cuda-11.8: /usr/local/cuda-11.8/
# torch: 2.0.1+cu118

# Usage: source dev/install_torch201_tensorrt.sh

set -e

export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.8/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH

mkdir -p tmp_env
hdfs dfs -get hdfs://hobot-bigdata/user/zhigang.yang/torch-tensorrt/torch_tensorrt-1.4.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl tmp_env/
hdfs dfs -get hdfs://hobot-bigdata/user/zhigang.yang/torch-tensorrt/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz tmp_env/

tar -xvf tmp_env/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz -C tmp_env/
export LD_LIBRARY_PATH=tmp_env/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib/:$LD_LIBRARY_PATH
export PATH=tmp_env/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/bin/:$PATH
pip3 install tmp_env/TensorRT-8.6.1.6/python/tensorrt-8.6.1-cp38-none-linux_x86_64.whl
pip3 install tmp_env/TensorRT-8.6.1.6/graphsurgeon/*.whl

pip3 install tmp_env/torch_tensorrt-1.4.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
