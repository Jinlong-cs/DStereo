#!/usr/bin/env bash

# This shell is used to install the env of torch-tensorrt.
# It should be noted that cuda-10.2 and cuda-11.1 must exist at the same time.
# Details reference to https://horizonrobotics.feishu.cn/wiki/wikcnrMtwgFz7wfeQGnzNQKWZie.

# cuda-10.2: /usr/local/cuda-10.2/
# cuda-11.1: /usr/local/cuda-11.1/
# torch: 1.10.2+cu111

# Usage: source dev/install_torch1102_tensorrt.sh

set -e

export PATH=/usr/local/cuda-11.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:/usr/local/cuda-11.1/targets/x86_64-linux/lib/:/usr/local/cuda-10.2/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH

mkdir -p tmp_env
hdfs dfs -get hdfs://hobot-bigdata/user/zhigang.yang/torch-tensorrt/torch_tensorrt-1.0.0-cp38-cp38-linux_x86_64.whl tmp_env/
hdfs dfs -get hdfs://hobot-bigdata/user/zhigang.yang/torch-tensorrt/TensorRT-8.0.1.6.Linux.x86_64-gnu.cuda-10.2.cudnn8.2.tar.gz tmp_env/

tar -xvf tmp_env/TensorRT-8.0.1.6.Linux.x86_64-gnu.cuda-10.2.cudnn8.2.tar.gz -C tmp_env/
export LD_LIBRARY_PATH=tmp_env/TensorRT-8.0.1.6/targets/x86_64-linux-gnu/lib/:$LD_LIBRARY_PATH
export PATH=tmp_env/TensorRT-8.0.1.6/targets/x86_64-linux-gnu/bin/:$PATH
pip3 install tmp_env/TensorRT-8.0.1.6/python/tensorrt-8.0.1.6-cp38-none-linux_x86_64.whl
pip3 install tmp_env/TensorRT-8.0.1.6/graphsurgeon/*.whl

pip3 install tmp_env/torch_tensorrt-1.0.0-cp38-cp38-linux_x86_64.whl
