from examples.classification.resnet18 import *

tensorrt_cfg = dict(
    model=deploy_model["backbone"],
    example_inputs=deploy_inputs["img"],
    input_cfg=dict(
        shape=[1, 3, 224, 224],
        dtype=torch.float32,
    ),
    compile_cfg=dict(
        enabled_precisions={torch.float32},
        truncate_long_and_double=True,
    ),
)
