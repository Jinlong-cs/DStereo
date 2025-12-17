# 使用 DCU 进行训练

目前 HAT 已完成了对 DCU 的适配，在用法上和 nvidia GPU 没有明显区别，但是在一些工具上可能不支持或者不同的用法，用户需要注意的主要有以下内容；

1. 查看 GPU 利用率及显存：GPU 是 nvidia-smi，在 DCU 中是 rocm-smi.
2. GPU 相关的工具不可用，如 HAT 提供的 gpu_affinity，PytorchProfiler 对 GPU 的 perf 等。
