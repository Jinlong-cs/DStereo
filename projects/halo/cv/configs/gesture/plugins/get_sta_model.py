from torch import nn

# 训练模式: 'train'; 部署分割模型: 'rgb' 'kps' 'head'; 动静结合: 'kps_rgb'.
mode = "train"

bn_kwargs = {
    "eps": 2e-5,
    "momentum": 0.9,
}

# for kps encoder
kps_backbone = dict(
    type="VargNetV2",
    num_classes=71,
    input_channels=3,
    include_top=False,
    alpha=0.5,
    bn_kwargs=bn_kwargs,  # default: eps 1e-5, m: 0.1
)

# for rgb encoder
rgb_backbone = dict(
    type="VargNetV2",
    num_classes=71,
    input_channels=3,
    include_top=False,
    alpha=0.5,
    bn_kwargs=bn_kwargs,  # default: eps 1e-5, m: 0.1
)

img_seq_len = 8
add_motion_in_rgb_branch = False
disable_quanti_input = False
enable_kps_gap = True


# for head
fusion_type = "result_fusion"
use_dropout = False
in_channels_dict = {
    "kps": 512,
    "frames": 512,
}
num_classes = 71
flat_output = True

# loss
fusion_type = "result_fusion"
loss_kps = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
loss_frames = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
loss_weight = [1, 1]


def get_sta_network(num_classes, deploy=False):
    if deploy:
        loss = None
    else:
        loss = dict(
            type="GestMultiModeLoss",
            fusion_type=fusion_type,
            loss_kps=loss_kps,
            loss_frames=loss_frames,
            loss_weight=loss_weight,
        )

    multimodality_classifier = dict(
        type="StaMultiModalityClassifier",
        kps_encoder=dict(
            type="ActKPSEncoder",
            backbone=kps_backbone,
            bn_kwargs=bn_kwargs,
        ),
        rgb_encoder=dict(
            type="ActRGBEncoder",
            backbone=rgb_backbone,
            img_seq_len=img_seq_len,
            bn_kwargs=bn_kwargs,
            add_motion_in_rgb_branch=add_motion_in_rgb_branch,
            disable_quanti_input=disable_quanti_input,
            mode=mode,
        ),
        head=dict(
            type="StaMultiModalityHead",
            fusion_type=fusion_type,
            use_dropout=use_dropout,
            in_channels_dict=in_channels_dict,
            num_classes=num_classes,
            flat_output=not deploy,
            bn_kwargs=bn_kwargs,
            add_motion_in_rgb_branch=add_motion_in_rgb_branch,
            mode=mode,
        ),
        loss=loss,
        mode=mode,
    )

    return multimodality_classifier
