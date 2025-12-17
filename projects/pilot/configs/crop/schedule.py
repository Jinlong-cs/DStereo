from common import pipeline_test

if pipeline_test:
    num_steps = dict(
        with_bn=5,
        freeze_bn_1=5,
        freeze_bn_2=5,
        freeze_bn_3=5,
        int_infer=0,
    )
    warmup_steps = 0
    save_interval = 5
else:
    num_steps = dict(
        with_bn=80000,
        freeze_bn_1=20000,
        freeze_bn_2=20000,
        freeze_bn_3=20000,
        int_infer=0,
    )
    warmup_steps = 1000
    save_interval = 1000

base_lr = dict(
    with_bn=0.0015,
    freeze_bn_1=0.00005,
    freeze_bn_2=0.00001,
    freeze_bn_3=0.00001,
    int_infer=0.0,
)

interval_by = "step"

train_stages = [
    "with_bn",
    "freeze_bn_1",
    "freeze_bn_2",
    "freeze_bn_3",
]


def get_fuse_patterns_by_stage(cur_stage):

    freeze_bn_modules = dict(
        with_bn=[],
        freeze_bn_1=[
            "^.*backbone",  # backbone
            "^.*fpn_neck",  # fpn
            "^.*fix_channel_neck",  # fix channel fpn
        ],
        freeze_bn_2=[
            "^.*ufpn.*neck",  # ufpn in seg and 3d tasks
            "^.*_anchor_head",  # rpn heads
        ],
        freeze_bn_3=[
            "^(?!.*anchor).*head",  # heads (other than anchor heads)
        ],
    )

    pre_fuse_patterns = []
    for stage, pattern in freeze_bn_modules.items():
        if stage == cur_stage:
            break
        pre_fuse_patterns += pattern

    cur_fuse_patterns = freeze_bn_modules[cur_stage]
    return pre_fuse_patterns, cur_fuse_patterns
