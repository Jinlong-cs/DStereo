from examples.classification.resnet18 import *

mem_profiler = dict(
    type="StageCPUMemoryProfiler",
    profile_action_name="perf_dataset",
    leaks=False,
    dirpath="./",
    filename="stage_cpu_profiler",
)

# perf memory
data_loader["dataset"] = dict(
    type="PerfDataset",
    dataset=data_loader["dataset"],
    profiler=mem_profiler,
    perf_interval=128,
)

python_profiler = dict(
    type="PythonProfiler",
    dirpath="work_dirs/hat_logss",
    filename="python_profiler",
)

# perf time cost
data_loader["dataset"] = dict(
    type="PerfDataset",
    dataset=data_loader["dataset"],
    profiler=python_profiler,
    perf_data_len=100,
)
