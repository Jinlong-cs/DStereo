import os

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
]


# is_local_train = not os.path.exists("/running_package")
is_local_train = True
expand_type = "longside_square_1.2"
if is_local_train:
    batch_size_per_gpu = 144
    device_ids = [0, 1, 2, 3]
    rec_prefix = "/horizon-bucket"
    num_workers = 0
    # ckpt_dir = "tmp_models"
    ckpt_dir = "/horizon-bucket/MultiMode_2/mm_algorithms_data/face-quality-attributes/hat_model/facequality_mtl"  # noqa
else:
    batch_size_per_gpu = 144
    device_ids = [0, 1, 2, 3]
    rec_prefix = "/bucket/output"
    num_workers = 4
    # ckpt_dir = "tmp_models"
    ckpt_dir = "/bucket/output/MultiMode_2/mm_algorithms_data/face-quality-attributes/hat_model/facequality_mtl"  # noqa
input_hw = (128, 128)

bn_kwargs = dict(eps=1e-5, momentum=0.1)


# common structures
backbone = dict(
    type="VargNetV2",
    num_classes=1000,
    model_type="tinyvargnetv2",
    include_top=False,
    bn_kwargs=bn_kwargs,
    node_name="backbone",
)


# transform
train_transforms = [
    dict(
        type="CropRecROI",
        crop_type="random",
        target_shape=(128, 128, 3),
        base_roi=[8, 8, 136, 136],
        crop_jitter_range=0.0625,
    ),
    # dict(
    #     type="RandomDownSample",
    #     p=0.3,
    #     inter_method=10),
    dict(
        type="RandomDownSample",
        p=0.3,
        data_shape=(3, 128, 128),
        min_downsample_width=128 * 1.5,
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
