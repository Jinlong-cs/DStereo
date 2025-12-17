from projects.pilot.configs.bev_7v.common import pipeline_test

if pipeline_test:
    num_steps = dict(
        float=10,
        float_freeze_bn=10,
        qat=10,
        int_infer=0,
        pack_infer=10,
    )
    warmup_steps = 0
    save_interval = 5
else:
    num_steps = dict(
        float=48000,
        float_freeze_bn=240000,
        qat=48000,
        int_infer=0,
        pack_infer=10,
    )
    warmup_steps = 4000
    save_interval = 2000

base_lr = dict(
    float=0.0005,
    float_freeze_bn=0.000375,
    qat=0.000025,
    int_infer=0.0,
    pack_infer=0.0,
)

interval_by = "step"

train_stages = [
    "float",
    "float_freeze_bn",
    "qat",
]
