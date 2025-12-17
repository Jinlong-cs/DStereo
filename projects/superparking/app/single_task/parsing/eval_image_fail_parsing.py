import os

import torch
import torch.nn.functional as F
from hatbc.filestream.bucket.client import BucketClient

from hat.utils.config import Config

this_dir = os.path.dirname(__file__)
cfg = Config.fromfile(this_dir + "/image_fail_parsing.py")

val_model = cfg.val_model
val_transforms = cfg.val_transforms
val_batch_size_per_gpu = 20
val_batch_processor = cfg.val_batch_processor
device_ids = [0]


def reformat_seg_to_aidi_eval(
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
    for img_name, objects, h_i, w_i in zip(
        batch_data["img_name"], batch_objects, h, w
    ):

        objects = objects.to("cpu")[
            None,
        ]
        objects = F.interpolate(
            objects.float(), size=(h_i, w_i), mode="bilinear"
        )
        objects = F.softmax(objects, dim=1).max(dim=1)[1][0]
        if os.path.splitext(img_name)[-1] != ".png":
            img_postfix = os.path.splitext(img_name)[-1]
            assert img_postfix in [".jpg", ".jpeg", "bmp"]
            img_name = img_name.replace(img_postfix, ".png")
        assert img_name.endswith(
            ".png"
        ), f"Image type error! expect .png file, but get {img_name}"
        ret = {
            "image_name": img_name,
            "out_img": objects.numpy(),
        }
        rets.append(ret)

    return rets


dataset_id_list = [
    "6031633",
    "6035920",
    "6037477",
]
bkt_clt = BucketClient()
local_datapath = bkt_clt.url_to_local(
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
        reformat_output_fn=reformat_seg_to_aidi_eval,
        reformat_out_fn_kwargs={"obj_key": "pred"},
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
        # dataset=val_dataset,
        batch_size=val_batch_size_per_gpu,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        drop_last=False,
    )
    val_data_loader.append(val_data_loader_i)

qat_pred = "http://fm-wangyi-shangguan.alitrain.hogpu.cc/plat_gpu/img_fail_master_test_no_remap-20221107-151441/output/qat-checkpoint-best-f77f5cec.pth.tar"

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
