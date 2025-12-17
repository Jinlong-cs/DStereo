from common import pipeline_test

if pipeline_test:
    num_steps = dict(
        float=10,
        float_freeze_bn=20,
        qat=100,
        int_infer=0,
    )
    warmup_steps = 0
    save_interval = 5
else:
    num_steps = dict(
        float=80000,
        float_freeze_bn=80000,
        qat=80000,
        int_infer=0,
    )
    warmup_steps = 2000
    save_interval = 5000

base_lr = dict(
    float=0.0003,
    float_freeze_bn=0.0003,
    qat=0.0001,
    int_infer=0.0,
)

interval_by = "step"

train_stages = [
    "float",
    "float_freeze_bn",
    "qat",
]


def get_fuse_patterns_by_stage(cur_stage):

    freeze_bn_modules = dict(
        float=[],
        float_freeze_bn=[],
        qat=[
            "^.*backbone",  # backbone
            "^.*fpn_neck",  # fpn
            "^.*fix_channel_neck",  # fix channel fpn
            "^.*ufpn.*neck",  # ufpn in seg and 3d tasks
            "^.*_anchor_head",  # rpn heads
            "^.*stage2_back_bone",  # bev stage2 backbone
            "^.*_stage1_head",  # bev stage1 heads
            "^.*_stage2_neck",  # bev stage2 necks
            "^.*_roi[^3]*head",  # roi head (without roi 3d head)
            "^(?!(.*roi|.*anchor|.*stage1)).*head",  # seg and 3d heads, bev stage2 heads  # noqa
        ],
    )

    pre_fuse_patterns = []
    for stage, pattern in freeze_bn_modules.items():
        if stage == cur_stage:
            break
        pre_fuse_patterns += pattern

    cur_fuse_patterns = freeze_bn_modules[cur_stage]
    return pre_fuse_patterns, cur_fuse_patterns
