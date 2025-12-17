from examples.classification.resnet18 import *

float_trainer["compiler"] = dict(
    type="Torch2Compile",
    fullgraph=False,
    dynamic=False,
    backend="inductor",
    mode="default",
)

profiler = dict(type="DynamoProfiler", filename="dynamo_result")
float_trainer["profiler"] = profiler
float_trainer["num_epochs"] = None
float_trainer["num_steps"] = 5  # just 5 steps
float_trainer["stop_by"] = "step"
float_trainer["callbacks"] = []
