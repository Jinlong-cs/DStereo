import copy
import os
import warnings

import torch
from horizon_plugin_pytorch.quantization import March

from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.utils.config import ConfigVersion

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"
root = os.path.join(bucket_root, "interaction")

if not os.path.isdir(root):
    raise FileNotFoundError("interaction bucket")

VERSION = ConfigVersion.v2
enable_model_tracking = True
enable_amp = False
warnings.filterwarnings("ignore")
NUM_LDMK = 68
LDMK_PAIRS = [
    [0, 16],
    [1, 15],
    [2, 14],
    [3, 13],
    [4, 12],
    [5, 11],
    [6, 10],
    [7, 9],
    [17, 26],
    [18, 25],
    [19, 24],
    [20, 23],
    [21, 22],
    [31, 35],
    [32, 34],
    [36, 45],
    [37, 44],
    [38, 43],
    [39, 42],
    [40, 47],
    [41, 46],
    [48, 54],
    [49, 53],
    [50, 52],
    [61, 63],
    [60, 64],
    [67, 65],
    [58, 56],
    [59, 55],
]
datasets_len = 2411563
bn_kwargs = dict(
    eps=2e-5, momentum=0.1
)  # because the different implement from pytorch and mxnet, momentum in gluon face is 0.9. # noqa


training_step = os.environ.get("HAT_TRAINING_STEP", "float")
task_name = "face_ldmk_vector_sid19"

if is_local_train:
    batch_size_per_gpu = 16
    device_ids = [0, 1, 2, 3]
    ckpt_dir = "./tmp_models/%s" % task_name
    if not os.path.isdir("tmp_models"):
        os.makedirs("./tmp_models")
else:
    batch_size_per_gpu = 32
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    ckpt_dir = "/job_data/models/%s" % task_name

num_epoch = 140
qat_epoch = 30
log_freq = 200
LR = 2e-3
STRIDE = 4
INPUT_H = 160
INPUT_W = 160
TARGET_H = INPUT_H // STRIDE
TARGET_W = INPUT_W // STRIDE
total_batch_size = batch_size_per_gpu * len(device_ids)
if datasets_len % total_batch_size == 0:
    total_steps = datasets_len // total_batch_size * num_epoch  # noqa
    qat_steps = datasets_len // total_batch_size * qat_epoch  # noqa
else:
    total_steps = (datasets_len // total_batch_size + 1) * num_epoch  # noqa
    qat_steps = (datasets_len // total_batch_size + 1) * qat_epoch  # noqa
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BAYES
qat_mode = "fuse_bn"

test_inputs = dict(img=torch.randn((1, 3, INPUT_H, INPUT_W)))

fpn_channel = 128
# decoder
decoder = dict(
    type="GroupFPN",
    feature_dim=fpn_channel,
    in_channels=[24, 48, 64, 128],  # strides:[4, 8, 16, 32]
    min_output_stride=4,
    group_base=16,
    bn_kwargs=bn_kwargs,
)

# mixvarenet config
mixvargenet_size = 0.75
net_config = [
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
        ),  # noqa
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
        ),  # noqa
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
        ),  # noqa
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
        ),  # noqa
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
        ),  # noqa
    ],  # stride 32
]

# Train Model
model = dict(
    type="LdmkModel",
    backbone=dict(
        type="MixVarGENet",
        net_config=net_config,
        num_classes=1000,
        bn_kwargs=bn_kwargs,
        include_top=False,
        bias=True,
    ),
    mode="train",
    decoder=decoder,
    vector_head=dict(
        type="LdmkVectorHead",
        in_channels=fpn_channel,
        num_ldmk=NUM_LDMK,
        band_width=1,
        vector_size=(TARGET_H, TARGET_W),
        band_module_type="conv",
        loss_func=dict(type="LdmkLoss", loss_type="l2"),
        bn_kwargs=bn_kwargs,
    ),
    coords_head=None,
    feat_stride=STRIDE,
    heatmap_head=None,
    cls_head=None,
    loss_weights={"vector": 1.0},
)

# Val Model
val_model = dict(
    type="LdmkModel",
    backbone=dict(
        type="MixVarGENet",
        net_config=net_config,
        num_classes=1000,
        bn_kwargs=bn_kwargs,
        include_top=False,
        bias=True,
    ),
    mode="val",
    decoder=decoder,
    vector_head=dict(
        type="LdmkVectorHead",
        in_channels=fpn_channel,
        num_ldmk=NUM_LDMK,
        band_width=1,
        vector_size=(TARGET_H, TARGET_W),
        band_module_type="conv",
        loss_func=dict(type="LdmkLoss", loss_type="l2"),
        bn_kwargs=bn_kwargs,
    ),
    coords_head=None,
    feat_stride=STRIDE,
    heatmap_head=None,
    cls_head=None,
    loss_weights={"vector": 1.0},
)

deploy_model = copy.deepcopy(val_model)
deploy_model["mode"] = "deploy"

deploy_inputs = dict(img=torch.randn((1, 3, INPUT_H, INPUT_W)))

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT"),
        dict(type="QAT2Quantize"),
    ],
)

# longside square dataset
train_recs = [
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_WFLW.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_tianjin201806.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_tianjin201806_plus_201808.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_tianjin201806_plus_201807.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_simrgb_office.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_simrgb_office_plus_201807.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_rgbinfra_downstairs.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_rgbinfra_downstairs_plus_201806.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_purchased_2nd_rgbinfra.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_purchased_1st_rgbinfra.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_purchased_1st_rgbinfra_20181127.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_purchased_1st_rgbinfra_20181126.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_pts72_refresh.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_pts72_refresh_plus_201807.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_njsmoke_20181102.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_pts72_refresh_plus_201808.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_njoffice201809.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_neimeng_ruilian_side_20181210.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_own_ruilian_smoke_20181212.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_neimeng_ruilian_front.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_nebulaV2_side1_wechat_20190325.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_nebulaV2_front_wechat_20190325.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_Menpo-3D.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_custom_sbao_yawn_20190701_02.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_custom_sbao_alarm_20190410.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_custom_alarm_20180615.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_carnet_realtest_20181129.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_AFLW2000-3D-Reannotated.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_custom_sbao_yawn_20190701_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_300W.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_300WLP.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_300VW-3D.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_300W-Testset-3D.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_Umd.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_IR2019_train_02.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_IR2019_train_03.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_IR2019_train_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_IR2019_train_00.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_gaze_68pts_train_04.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_gaze_68pts_train_02.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_gaze_68pts_train_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_gaze_68pts_train_00.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_gaze_68pts_train_03.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_68pts_guangqi_yawn_train.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_68pts_smoke_train.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_child_68pts_train_00.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_children_68pts_train_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_children_68pts_train_02.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_children_68pts_train_03.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_children_68pts_train_04.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_children_68pts_train_05.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_s202da_train_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_train_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_train_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_train_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_train_02.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_train_02.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_train_02.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_train_03.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_train_03.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_train_03.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_02.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_02.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_02.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_03.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_03.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_03.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_04.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_04.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_04.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_05.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_05.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_05.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_06.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_06.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_06.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_07.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_07.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_07.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_08.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_08.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_08.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_09.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_09.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_09.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_10.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_10.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_10.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_11.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_11.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_11.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_12.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_12.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_12.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_13.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_13.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_13.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_14.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_14.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_14.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_15.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_15.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_15.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_16.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_16.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_16.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_17.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_17.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_17.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_18.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_18.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_18.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_19.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_19.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_19.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_20.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_20.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_20.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_21.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_21.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_21.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_22.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_22.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_22.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_23.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_23.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_23.rec",  # noqa
]

val_recs = [
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks68_rgb_test.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks68_ir_test.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_test_01.rec",  # noqa
    # f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_largepose_test.rec", # noqa
    # f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_rgb_68pts_yawn_test.rec", # noqa
    # f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_yawn_test.rec", # noqa
    # f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_phone_68pts_test_01.rec", # noqa
    # f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_rgb_68pts_mask_test.rec", # noqa
    # f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_gaze_test.rec", # noqa
    # f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_emotion_test.rec", # noqa
    # f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_children_68pts_test_01.rec", # noqa
    # f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_68pts_sunglasses_test.rec", # noqa
    # f"{root}/face/landmark/68pts/rec/rec_long_square/lmks68_rgb_test_clockwise60.rec", # noqa
    # f"{root}/face/landmark/68pts/rec/rec_long_square/lmks68_rgb_test_clockwise30.rec", # noqa
]

val_data_descs = [
    "test_lmks_pose_Lmks68RGBTest_lmks68",
    "test_lmks_pose_Lmks68IRTest_lmks68",
    "lmks_large_pose_68pts_test_01",
    # "lmks_ir_68pts_largepose_test",
    # "lmks_rgb_68pts_yawn_test",
    # "lmks_ir_68pts_yawn_test",
    # "lmks_ir_phone_68pts_test_01",
    # "lmks_rgb_68pts_mask_test",
    # "lmks_ir_68pts_gaze_test",
    # "lmks_ir_68pts_emotion_test",
    # "lmks_children_68pts_test_01",
    # "lmks_68pts_sunglasses_test",
    # "lmks68_rgb_test_clockwise60",
    # "lmks68_rgb_test_clockwise30",
]


# transform
train_transforms = [
    dict(
        type="CropRecROI",
        crop_type="random",
        target_shape=(INPUT_H, INPUT_W, 3),
        base_roi=[51, 51, 205, 205],  # ratio 1.2
        crop_jitter_range=0.2,
        random_type="gaussian",
    ),
    dict(
        type="RandomFlip",
        px=0.5,
    ),
    dict(
        type="RandomShiftRotateScale",
        rotate_prob=0.5,
        bounded=True,
        max_rotate_angle=30,
        resize=True,
    ),
    dict(
        type="GaussianNoise",
        prob=0.2,
        mean=0,
        sigma=2,
    ),
    dict(
        type="RandomNoise",
        prob=0.2,
        min=-5,
        max=5,
    ),
    dict(
        type="SaltPepperNoise",
        prob=0.2,
        s_ratio=0.05,
        p_ratio=0.05,
    ),
    dict(
        type="GaussianBlur",
        p=0.2,
        kernel_size_min=3,
        kernel_size_max=9,
        sigma_min=1.0,
        sigma_max=5.0,
    ),
    dict(
        type="MotionBlur",
        p=0.2,
        length_min=2,
        length_max=10,
        angle_min=1,
        angle_max=359,
    ),
    dict(
        type="RandomGray",
        p=0.3,
    ),
    dict(
        type="RandomOcclusion",
        prob=0.5,
        occ_type="whole",
        occ_num=1,
        size_ratio=0.7,
    ),
    dict(
        type="GenerateGaussianVector",
        num_ldmk=NUM_LDMK,
        feat_stride=STRIDE,
        vector_size=(TARGET_H, TARGET_W),
        sigma=2,
        encoding_method="standard",
    ),
    dict(type="ToTensor"),
]
val_transforms = [
    dict(
        type="CropRecROI",
        crop_type="center",
        target_shape=(INPUT_H, INPUT_W, 3),
        base_roi=[51, 51, 205, 205],
        crop_jitter_range=0.0,
    ),
    dict(
        type="GenerateGaussianVector",
        num_ldmk=NUM_LDMK,
        feat_stride=STRIDE,
        vector_size=(TARGET_H, TARGET_W),
        sigma=2,
        encoding_method="standard",
    ),
    dict(type="ToTensor"),
]


# datasets
train_datasets = dict(
    type="LdmkDataset",
    rec_list=train_recs,
    num_ldmk=NUM_LDMK,
    use_3d=False,
    task_type="face",
    data_type="rec",
    transforms=train_transforms,
    ldmk_pairs=LDMK_PAIRS,
)

val_datasets = dict(
    type="LdmkDataset",
    rec_list=val_recs,
    data_desc=val_data_descs,
    num_ldmk=NUM_LDMK,
    use_3d=False,
    task_type="face",
    data_type="rec",
    transforms=val_transforms,
    ldmk_pairs=LDMK_PAIRS,
)


# dataloader
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=train_datasets,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
)


val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=val_datasets,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=16,
    pin_memory=True,
)

batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=[
        dict(
            type="Lighting",
            prob=1.0,
            alphastd=0.2,
        ),
        dict(
            type="TorchVisionAdapter",
            interface="ColorJitter",
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0,
        ),
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    loss_collector=collect_loss_by_regex("total_loss"),
    enable_amp=enable_amp,
)
val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    batch_transforms=[
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
)


def update_metric(metrics, batch, model_outs):
    for metric, key in zip(metrics, model_outs):
        metric.update(model_outs[key])


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
)


def update_val_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


def get_update_metric(data_desc):
    def _update(metrics, _batch, _model_outs):
        batch = copy.deepcopy(_batch)
        model_outs = copy.deepcopy(_model_outs)
        data_descs = batch["data_desc"]
        gt_ldmk = model_outs["gt_ldmk"]
        pr_x = model_outs["pr_vector_x"]
        pr_y = model_outs["pr_vector_y"]
        imgs = _batch["img"]
        img_list = []
        ldmk_list = []
        pr_x_list = []
        pr_y_list = []
        for i in range(len(data_descs)):
            if data_desc == data_descs[i]:
                img_list.append(imgs[i])
                ldmk_list.append(gt_ldmk[i])
                pr_x_list.append(pr_x[i])
                pr_y_list.append(pr_y[i])
        if len(ldmk_list) != 0:
            data = {
                "img": torch.stack(img_list),
                "gt_ldmk": torch.stack(ldmk_list),
                "pr_vector_x": torch.stack(pr_x_list),
                "pr_vector_y": torch.stack(pr_y_list),
            }
            for metric in metrics:
                metric.update(data)

    return _update


val_metric_updater_list = [
    dict(
        type="MetricUpdater",
        log_prefix=f"Validation {data_desc_i}\n",
        metrics=[
            dict(
                type="NormalizedMeanError",
                num_ldmk=NUM_LDMK,
                norm_type="ION",
                mode="vector",
                decoding_method="diff",
                feat_stride=4,
                name="LdmkION",
            )
        ],
        metric_update_func=get_update_metric(data_desc_i),
        step_log_freq=-1,
        epoch_log_freq=1,
        reset_metrics_by="epoch",
    )
    for data_desc_i in val_data_descs
]

val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_val_metric,
    step_log_freq=-1,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=log_freq,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    save_interval=1,
    interval_by="epoch",
    strict_match=False,
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=val_metric_updater_list,
    interval_by="epoch",
    val_interval=1,
    val_model=val_model,
    val_on_train_end=True,
    log_interval=10,
)

qat_val_callback = copy.deepcopy(val_callback)
qat_val_callback.update(
    dict(
        model_convert_pipeline=dict(
            type="ModelConvertPipeline",
            converters=[
                dict(type="Float2QAT"),
            ],
        )
    )
)


aidi_exp_model_callback = dict(
    type="AIDIExpModel",
    model_name=task_name,
    task_type="classification",
    save_model="best",
    platforms=["J5"],
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=1e-4)},
        lr=LR,
    ),
    batch_processor=batch_processor,
    # num_epochs=num_epoch,
    # stop_by="epoch",
    num_steps=500,
    stop_by="step",
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="PolyLrUpdater",
            max_update=total_steps,
            step_log_interval=1000,
            power=2,
            final_lr=1e-6,
            warmup_by="step",
            warmup_len=1000,
            warmup_begin_lr=0,
            warmup_mode="linear",
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow", name="LdmkLoss"),
        dict(type="LossShow", name="Loss"),
    ],
)


qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
                allow_miss=False,
                ignore_extra=True,
                verbose=True,  # Show unexpect_key and miss_key info.
            ),
            dict(type="Float2QAT"),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=4e-5)},
        lr=2e-6,
    ),
    batch_processor=batch_processor,
    num_epochs=qat_epoch,
    stop_by="epoch",
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="PolyLrUpdater",
            max_update=qat_steps,
            step_log_interval=1000,
            power=2,
            final_lr=1e-7,
        ),
        # dict(
        #     type="StepDecayLrUpdater",
        #     lr_decay_id=[15, 22],
        #     lr_decay_factor=0.1,
        #     step_log_interval=1000,
        # ),
        # dict(
        #     type="CosLrUpdater",
        #     step_log_interval=1000,
        # ),
        metric_updater,
        qat_val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow", name="LdmkLoss"),
        dict(type="LossShow", name="Loss"),
    ],
)


int_infer_trainer = dict(
    type="Trainer",
    model=deploy_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
                allow_miss=False,
                ignore_extra=True,
                verbose=True,  # Show unexpect_key and miss_key info.
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
    callbacks=[
        ckpt_callback,
        trace_callback,
    ],
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid"],
)

# predictor
float_predictor = dict(
    type="Predictor",
    model=val_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
                allow_miss=False,
                ignore_extra=True,
                verbose=True,  # Show unexpect_key and miss_key info.
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    callbacks=val_metric_updater_list,
)

qat_predictor = dict(
    type="Predictor",
    model=copy.deepcopy(val_model),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
                allow_miss=False,
                ignore_extra=True,
                verbose=True,  # Show unexpect_key and miss_key info.
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    callbacks=val_metric_updater_list,
)

int_infer_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    callbacks=val_metric_updater_list,
)

# onnx
onnx_cfg = dict(
    model=deploy_model,
    stage="float",
    inputs=deploy_inputs,
    model_convert_pipeline=float_predictor["model_convert_pipeline"],
)
