from projects.pilot.configs.bev_7v_temporal.common import (
    is_local_train,
    pipeline_test,
)

if pipeline_test or is_local_train:
    num_steps = dict(
        float=10,
        float_freeze_bn=10,
        # calibration=1,
        qat=10,
        int_infer=0,
        pack_infer=0,
    )
    warmup_steps = 0
    save_interval = 5
else:
    num_steps = dict(
        float=48000,
        float_freeze_bn=320000,
        # calibration=250,
        qat=48000,
        int_infer=0,
        pack_infer=0,
    )
    warmup_steps = 4000
    save_interval = 1000

base_lr = dict(
    float=0.0005,
    float_freeze_bn=0.00008,
    # calibration=0.0,
    qat=0.000008,
    int_infer=0.0,
    pack_infer=0.0,
)

interval_by = "step"

train_stages = [
    "float",
    "float_freeze_bn",
    # "calibration",
    "qat",
]
