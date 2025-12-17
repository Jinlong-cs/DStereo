import os

debug_mode = False
training_step = os.getenv("HAT_TRAINING_STEP", "float")
remote_pipeline_test = os.getenv("HAT_REMOTE_PIPELINE_TEST", "0") == "1"
pipeline_test = (
    os.getenv("HAT_PIPELINE_TEST", "0") == "1" or remote_pipeline_test
)
assert training_step in ["float", "freeze_bn", "qat", "int_infer"]

local_train = not os.path.exists("/running_package")
task_name = "vehicle_side"
input_hw = (192, 960)

# backbone
bn_kwargs = dict(eps=1e-5, momentum=0.1)
vargnetv2_2631_backbone = dict(
    type="VargNetV2Stage2631",
    num_classes=1000,
    multiplier=0.5,
    group_base=8,
    last_channels=1024,
    stages=(1, 2, 3, 4, 5),
    use_bias=True,
    include_top=False,
    extend_features=True,
    bn_kwargs=bn_kwargs,
    dropout=dict(p=0.3),
)
# neck
stride2channels = {
    2: 16,
    4: 16,
    8: 32,
    16: 64,
    32: 128,
    64: 128,
    128: 256,
    256: 512,
}
out_stride2channels = stride2channels
in_strides = [2, 4, 8, 16, 32, 64]
out_strides = [4, 8, 16, 32]
mono_unet_neck = dict(
    type="Unet",
    in_strides=in_strides,
    out_strides=out_strides,
    stride2channels=stride2channels,
    out_stride2channels=out_stride2channels,
    group_base=8,
)
# the number of hidden channels
feat_channels = 32

batch_size_factor = 0.5 if (not remote_pipeline_test) and pipeline_test else 1
base_step = 100 if remote_pipeline_test else 1 if pipeline_test else 50000
# trainer (for 1 machine 8 gpus case)
lr = 0.001 * batch_size_factor
freeze_bn_lr = 0.0002 * batch_size_factor
qat_lr = 0.0002 * batch_size_factor
wd = 1e-05
log_freq = int((200 / batch_size_factor + 1))
float_steps = int((base_step / batch_size_factor + 1))
freeze_bn_steps = int((base_step / 5 / batch_size_factor + 1))
qat_steps = int((base_step / batch_size_factor + 1))
warmup_steps = int((base_step / 50 / batch_size_factor + 1))
save_interval = int((base_step / 5 / batch_size_factor + 1))
interval_by = "step"
local_save_prefix = "tmp_output"
remote_save_prefix = "/job_data/models/"
save_prefix = local_save_prefix if local_train else remote_save_prefix
ckpt_dir = os.path.join(save_prefix, task_name)
log_dir = os.path.join(ckpt_dir, "logs")

if input_hw == (192, 960):
    global_desc = dict(
        roi_input=dict(
            fp_x=960 // 2,
            fp_y=540 // 2 - 220,
            width=960,
            height=192,
        ),
        vanishing_point=[960 // 2, 540 // 2],
    )
elif input_hw == (256, 960):
    global_desc = dict(
        roi_input=dict(
            fp_x=960 // 2,
            fp_y=540 // 2 - 188,
            width=960,
            height=256,
        ),
        vanishing_point=[960 // 2, 540 // 2],
    )
else:
    raise NotImplementedError
