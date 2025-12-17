# HAT 环境变量

## HAT 内部变量

内部变量为HAT内部自定义的环境变量

|       os.environ         |       说明         |   常见的值   |
| ---------------------    | :----------------: | :-------: |
|  "HAT_TRAINING_STEP"     |     当前阶段        | "float"、"qat"、"int_infer" |
|  "HAT_INFERENCE_STEP"    |     当前阶段        | "float"、"qat"、"int_infer" |
|  "HAT_PIPELINE_TEST"     | 是否测试整个pipeline |  True or False              |
|  "HAT_USE_CHECKPOINT"    | 是否使用checkpoint模式 |  True or False              |
|  "HAT_USE_SAVEDTENSOR"   | 是否使用savedtensor模式 |  True or False              |
|  "HAT_ENABLE_MODEL_TRACKING"   | 是否使用AIDI Model Tracking |  True or False              |
|  "HAT_PROCESS_GROUP_TIMEOUT"  | process group's timeout(seconds) |  default = 1800        |
|  "OPENCV_NUM_THREADS"    |  opencv 的线程数      | default = 12 |
|  "TORCH_NUM_THREADS"      |  torch 的线程数        | default = 12 |
|  "HAT_DETERMINISTIC_LEVEL"      |  non-deterministic op、module 处理 level  | "0", "1", "2" |


## 系统变量

|   os.environ            | 说明 |
| ---------------------| :--------------: |
|  "CUDA_VISIBLE_DEVICES" |  可见的gpus |
|  "OPENBLAS_NUM_THREADS" |  openblas 环境变量, default =12 |
|  "OMP_NUM_THREADS" |  openmp 环境变量, default =12 |
|  "MKL_NUM_THREADS" |  mkl 环境变量, default =12 |
|   "NCCL_DEBUG"     |  控制从 NCCL 显示的调试信息, default="INFO" |



## Pytorch 环境变量
具体见 [`Pytorch 环境变量`](https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/distributed.py)。
