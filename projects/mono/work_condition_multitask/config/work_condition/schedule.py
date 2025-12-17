from common import pipeline_test

if pipeline_test:
    num_steps = dict(
        with_bn=16,
        freeze_bn_1=16,
        freeze_bn_2=16,
        int_infer=0,
    )
    warmup_steps = 0
    save_interval = 5
else:
    num_steps = dict(
        with_bn=80000,
        freeze_bn_1=20000,
        freeze_bn_2=20000,
        int_infer=0,
    )
    warmup_steps = 1000
    save_interval = 100

base_lr = dict(
    with_bn=0.0015,
    freeze_bn_1=0.00001,
    freeze_bn_2=0.000005,
    int_infer=0.0,
)

interval_by = "step"

train_stages = [
    "with_bn",
    "freeze_bn_1",
    "freeze_bn_2",
]


def get_fuse_patterns_by_stage(cur_stage):

    freeze_bn_modules = dict(
        with_bn=[],
        freeze_bn_1=["^.*backbone", "^.*neck", "^.*extra_layers"],  # backbone
        freeze_bn_2=[
            "^.*head",
        ],
    )

    pre_fuse_patterns = []
    for stage, pattern in freeze_bn_modules.items():
        if stage == cur_stage:
            break
        pre_fuse_patterns += pattern

    cur_fuse_patterns = freeze_bn_modules[cur_stage]
    return pre_fuse_patterns, cur_fuse_patterns
