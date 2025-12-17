import os

from hat.models.backbones.mixvargenet import MixVarGENetConfig

# env variables
training_step = os.environ.get("HAT_TRAINING_STEP", "float")


# model info
# model_type = "facequality_multitask"
# model_name = "_".join([model_type, model_setting, model_version])


# tasks
tasks = [
    dict(name="blur"),
    dict(name="brightness"),
    dict(name="leye"),
    dict(name="reye"),
    dict(name="forehead"),
    dict(name="mouth"),
    dict(name="mask"),
    dict(name="glass"),
    dict(name="cap"),
    dict(name="faceldmk68"),
    dict(name="face3d"),
]

task_to_module = {
    "blur": "facequality.blur",
    "brightness": "facequality.brightness",
    "leye": "facequality.leye",
    "reye": "facequality.reye",
    "forehead": "facequality.forehead",
    "mouth": "facequality.mouth",
    "mask": "facequality.mask",
    "glass": "facequality.glass",
    "cap": "facequality.cap",
    "faceldmk68": "faceldmk.faceldmk68",
    "face3d": "face3d.face3d",
}

# parameters
is_local_train = not os.path.exists("/running_package")

# facequality expand_type
# expand_type = "longside_square_1.2"          # mixvargenet1.0 trainset & testset
expand_type = "longside_square_1.2_size160"  # mixvargenet0.75 testset
# expand_type = "longside_square_size256"      # mixvargenet0.75 trainset
step_log_freq = 500
dynamic_weighted_loss = False
if is_local_train:
    device_ids = [3]

    batch_size_per_gpu_facequality = 64
    batch_size_per_gpu_ldmk = 32
    batch_size_per_gpu_face3d = 20

    num_workers = 1
    rec_prefix = "/horizon-bucket"
    ckpt_dir = "/horizon-bucket/MultiMode_2/mm_algorithms_data/face-quality-attributes/hat_model/face_mtl"  # noqa
else:
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    # 64 for 3090
    # 32 for 2080ti
    batch_size_per_gpu_facequality = 64
    # 64 for 3090
    # 32 for 2080ti
    batch_size_per_gpu_ldmk = 64
    # 60 for 3090
    # 20 for 2080ti
    batch_size_per_gpu_face3d = 60

    num_workers = 2
    rec_prefix = "/bucket/output"
    # ckpt_dir = "/bucket/output/MultiMode_2/mm_algorithms_data/face-quality-attributes/hat_model/face_mtl"  # noqa
    ckpt_dir = "/job_data/models"
bn_kwargs = dict(eps=1e-5, momentum=0.1)

backbone_type = "mixvargenet0.75"
ldmk_in_channel = 120
in_channels = 320
input_hw = (128, 128)
if backbone_type == "vargnet1.25":
    in_channels = 320
elif backbone_type == "vargnet1.0":
    in_channels = 256
elif backbone_type == "mixvargenet0.75":
    in_channels = 128
    input_hw = (160, 160)
elif backbone_type == "mixvargenet1.0":
    in_channels = 160

loss_weights = {
    "blur": 36.0,
    "brightness": 32.0,
    "leye": 32.0,
    "reye": 32.0,
    "forehead": 16.0,
    "mouth": 16.0,
    "mask": 16.0,
    "glass": 144.0,
    "hat": 16.0,
    "face_ldmk": 1.0 / batch_size_per_gpu_ldmk,
    "face3d": 20.0,
}

# loss_weights = {
#     "blur": 1.0,
#     "brightness": 1.0,
#     "leye": 1.0,
#     "reye": 1.0,
#     "forehead": 1.0,
#     "mouth": 1.0,
#     "mask": 1.0,
#     "glass": 1.0,
#     "hat": 1.0,
#     "face_ldmk": 1.0,
#     "face3d_finetune": 1.0,
# }


# src mixvargnet 0.75
src_mixvargenet_075_config = [
    [
        MixVarGENetConfig(
            in_channels=24,
            out_channels=24,
            head_op="mixvarge_f2",
            stack_ops=[],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 2
    [
        MixVarGENetConfig(
            in_channels=24,
            out_channels=24,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 4
    [
        MixVarGENetConfig(
            in_channels=24,
            out_channels=48,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 8
    [
        MixVarGENetConfig(
            in_channels=48,
            out_channels=64,
            head_op="mixvarge_f2_gb16",
            stack_ops=[
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
            ],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 16
    [
        MixVarGENetConfig(
            in_channels=64,
            out_channels=128,
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 32
]

# src_mixvargenet1.0
src_mixvargenet_10_config = [
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=32,
            head_op="mixvarge_f2",
            stack_ops=[],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 2
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=32,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 4
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=64,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 8
    [
        MixVarGENetConfig(
            in_channels=64,
            out_channels=96,
            head_op="mixvarge_f2_gb16",
            stack_ops=[
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
            ],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 16
    [
        MixVarGENetConfig(
            in_channels=96,
            out_channels=160,
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 32
]


backbone = None
if "mixvargenet" in backbone_type:
    if backbone_type == "mixvargenet0.75":
        net_config = src_mixvargenet_075_config
    elif backbone_type == "mixvargenet1.0":
        net_config = src_mixvargenet_10_config
    # mixvargenet
    backbone = dict(
        type="MixVarGENet",
        net_config=net_config,
        input_channels=3,
        num_classes=1000,
        bn_kwargs=bn_kwargs,
        include_top=False,
        bias=True,
        node_name="facemtl_backbone",
    )
elif "vargnet" in backbone_type:
    # vargnet1.25
    backbone = dict(
        type="VargNetV2",
        alpha=1.25,
        num_classes=1000,
        include_top=False,
        group_base=8,
        # group_base=4,  # face ldmk single task setting
        factor=2,
        bn_kwargs=bn_kwargs,
        node_name="facemtl_backbone",
    )


# transform
train_transforms = [
    dict(
        type="CropRecROI",
        crop_type="random",
        # 128*128
        # target_shape=(128, 128, 3),
        # base_roi=[8, 8, 136, 136],
        # crop_jitter_range=0.0625,
        # 160*160 2t=256 => t=128 => 1.2t=154 => 1.2t resize to 160 == tesetset
        target_shape=(160, 160, 3),
        base_roi=[51, 51, 205, 205],
        crop_jitter_range=0.1,
    ),
    # dict(
    #     type="RandomDownSample",
    #     p=0.3,
    #     inter_method=10),
    dict(
        type="RandomDownSample",
        p=0.3,
        # 128*128
        # data_shape=(3, 128, 128),
        # min_downsample_width=128 * 1.5,
        # 160*160
        data_shape=(3, 160, 160),
        min_downsample_width=160 * 1.5,
        inter_method=10,
    ),
    dict(
        type="RandomShiftRotateScale",
        rotate_prob=0.3,
        max_rotate_angle=20,
        resize=True,
    ),
    dict(type="ToTensor"),
    dict(type="FaceQualityTransformLabel"),
]

batch_transforms = [
    dict(type="BgrToYuv444", rgb_input=True),
    dict(
        type="TorchVisionAdapter",
        interface="Normalize",
        mean=128.0,
        std=128.0,
    ),
]

test_transforms = [
    dict(type="ToTensor"),
    dict(type="FaceQualityTransformLabel"),
]
