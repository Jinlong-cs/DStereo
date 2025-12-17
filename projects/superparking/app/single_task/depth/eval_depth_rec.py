import os

import numpy as np
import torch
import torch.nn.functional as F
from hatbc.filestream.bucket.client import BucketClient

from hat.utils.config import Config

this_dir = os.path.dirname(__file__)
cfg = Config.fromfile(this_dir + "/depth_rec.py")
virtual_cam_params = np.array(
    [[110, 0, 176], [0, 110, 144], [0, 0, 1]], dtype=np.float32
)
val_model = cfg.val_model
fake_gt_depth = np.full([288, 352], (cfg.low_depth + cfg.high_depth) / 2)
val_model["target_generator"]["downsample_scale"] = 1
val_transforms = [
    dict(type="YUVTurboJPEGDecoder", to_string=True),
    # image in eval platform with size hxw 288 x 352, no need to resize
    dict(type="BPUPyramidResizer", scale_wh=[1, 1], pyramid_type="ips"),
    dict(type="ImgBufToYUV444"),
    dict(
        type="AddKeys", kv={"gt_depth": fake_gt_depth}
    ),  # fake depth, needed by forward
    dict(type="AddKeys", kv={"virtual_cam_params": virtual_cam_params}),
    dict(type="ToTensor", to_yuv=False),
    dict(type="Normalize", mean=128.0, std=128.0),
    dict(type="RenameKeys", keys=["imgs|img"]),
]
val_batch_size_per_gpu = 20
val_batch_processor = cfg.val_batch_processor
device_ids = [0]


def reformat_depth_to_aidi_eval(
    batch_data,
    batch_outputs,
    obj_key,
):
    if isinstance(batch_data, tuple):
        assert isinstance(
            batch_data[1], str
        ), f"Batch format error, it should be (batch, object_name), \
             but {batch_data}"
        batch_data, batch_obj_key = batch_data
    batch_objects = batch_outputs[obj_key]
    rets = []
    h, w = batch_data["transform_meta"][0]["origin_hw"]
    valid_mask = batch_data["mask"]
    for img_name, objects, h_i, w_i, mask_i in zip(
        batch_data["img_name"], batch_objects, h, w, valid_mask
    ):
        objects[0:1][
            ~mask_i
        ] = 0.0  # fov filter, not useful, need to change gt but not pred
        objects = objects.to("cpu")[
            None,
        ]
        objects = F.interpolate(
            objects.float(), size=(h_i, w_i), mode="bilinear"
        )[0]
        depth, conf = torch.split(objects, 1, dim=0)
        depth = depth * 256
        depth = depth.cpu().numpy().astype("uint16")
        conf = conf * 256
        conf = conf.cpu().numpy().astype("uint16")
        if os.path.splitext(img_name)[-1] != ".png":
            img_postfix = os.path.splitext(img_name)[-1]
            assert img_postfix in [".jpg", ".jpeg", "bmp"]
            img_name = img_name.replace(img_postfix, ".png")
        assert img_name.endswith(
            ".png"
        ), f"Image type error! expect .png file, but get {img_name}"
        ret = {
            "image_name": img_name,
            "depth": depth[0],
            "conf": conf[0],
        }
        rets.append(ret)

    return rets


dataset_id_list = [
    "6037567",
    "6037580",
]

local_datapath = BucketClient().url_to_local(
    "dmpv2://auto_eval/adas_eval/eval_platform/fs/"
)
dataset_val_img_path = [
    os.path.join(local_datapath, str(dataset_id), "datasets")
    for dataset_id in dataset_id_list
]

aidi_eval_callbacks = []
for i in dataset_id_list:
    aidi_eval_callback = dict(
        type="AIDIEval",
        prediction_name="test",
        output_root="tmp_output/prediction",
        project_id="PDT2021002",
        aidi_eval_dataset_id=[int(i)],
        reformat_output_fn=reformat_depth_to_aidi_eval,
        reformat_out_fn_kwargs={"obj_key": "depth_preds"},
    )
    aidi_eval_callbacks.append(aidi_eval_callback)
val_data_loader = []
for i in dataset_val_img_path:
    val_data_loader_i = dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="Auto2dFromImage",
            data_path=i,
            to_rgb=True,
            transforms=val_transforms,
        ),
        batch_size=val_batch_size_per_gpu,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        drop_last=False,
    )
    val_data_loader.append(val_data_loader_i)

# For your information: need to turn off relu6 check for this checkpoint file
qat_pred = "dmpv2://SuperParking/yunfeng.zhang/model_tmp/depth/v0.6.0/qat-checkpoint-last-efa2b13d.pth.tar"

int_predictor = dict(
    type="Predictor",
    model=val_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=qat_pred,
                allow_miss=True,
                ignore_extra=True,
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    callbacks=aidi_eval_callbacks,
    share_callbacks=False,
    log_interval=50,
)
