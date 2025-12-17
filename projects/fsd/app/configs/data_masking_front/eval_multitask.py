import os
from glob import glob

import torch
from common import (
    batch_size_per_gpu,
    bucket_root,
    log_freq,
    num_workers,
    pin_memory,
    save_prefix,
)
from models import val_model

from hat.utils.apply_func import convert_numpy

local_eval = True
callbacks = []
data_loaders = []
prediction_name = "sd_datamasking_fcos"
prediction_tags = ["superdrive", "resize"]

dataset_ids = dict(
    face_detection=[6040715],
    vehicle_plate_detection=[6040434],
)

transforms = [
    dict(
        type="Resize",
        img_scale=(540, 960),
    ),
    dict(
        type="FixedCrop",
        size=(0, 0, 960, 512),
    ),
    dict(type="ToTensor", to_yuv=True),
    dict(type="Normalize", mean=128.0, std=128.0),
]


def reformat_output_fn(batch, model_outs, task_name):
    if isinstance(batch, tuple):
        assert isinstance(
            batch[1], str
        ), f"Batch format error, it should be (batch, object_name), \
             but {batch}"
        batch, batch_obj_key = batch
    model_outs = model_outs[batch_obj_key][0]["pred_boxes"]
    rets = []
    batch = convert_numpy(batch)

    for img_name, scale_factor, model_out in zip(
        batch["img_name"], batch["scale_factor"], model_outs
    ):
        model_out = model_out.to("cpu")
        boxes = model_out.boxes.numpy()
        scores = model_out.scores.numpy()
        boxes = boxes / scale_factor
        dets = []
        for box, score in zip(boxes, scores):
            dets.append(
                {
                    "bbox": box.tolist(),
                    "bbox_score": score.tolist(),
                    "attrs": {},
                }
            )
        rets.append({"image_key": img_name, task_name: dets})
    return rets


for task, ids in dataset_ids.items():
    for id in ids:
        output_root = os.path.join(save_prefix, "eval", str(id))
        os.makedirs(output_root, exist_ok=True)
        task_name = "face" if task == "face_detection" else "plate"
        if local_eval:
            callback = dict(
                type="LocalEval",
                output_root=output_root,
                reformat_output_fn=reformat_output_fn,
                reformat_out_fn_kwargs=dict(
                    task_name="face" if task == "face_detection" else "plate"
                ),
                task_name="Detection2D",
                eval_handler_kwargs=dict(
                    gt_file=glob(
                        os.path.join(
                            f"/horizon-bucket/auto_eval/adas_eval/eval_platform/fs/{id}/datasets/*",  # noqa
                            "data.json",
                        )
                    )[0],
                    setting_file=os.path.join(
                        os.path.dirname(__file__),
                        f"settings/{task_name}_minH32_iou0.5_roi.yaml",
                    ),
                ),
            )
        else:
            callback = dict(
                type="AIDIEval",
                project_id="PDT2021004-vision",
                aidi_eval_dataset_id=id,
                output_root=output_root,
                prediction_name=prediction_name,
                prediction_tags=prediction_tags,
                reformat_output_fn=reformat_output_fn,
                reformat_out_fn_kwargs=dict(task_name=task_name),
            )

        data_loader = dict(
            type=torch.utils.data.DataLoader,
            sampler=dict(
                type=torch.utils.data.DistributedSampler,
                shuffle=False,
            ),
            dataset=dict(
                type="Auto2dFromImage",
                data_path=os.path.join(
                    bucket_root,
                    "auto_eval/adas_eval/eval_platform/fs/",
                    str(id),
                    "datasets",
                ),
                to_rgb=True,
                return_img_buf=False,
                return_orig_img=False,
                transforms=transforms,
            ),
            batch_size=batch_size_per_gpu * 4,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        callbacks.append(callback)
        data_loaders.append(
            dict(
                type="MultitaskLoader",
                return_task=True,
                loaders={task: data_loader},
            )
        )


int_infer_predictor = dict(
    type="Predictor",
    model=val_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path="http://fm-xiang-yan.bcloud-2nd.hogpu.cc/plat_gpu/hobot-dag-3411528_hat-superdrive-data-masking-front-20230714-144808/output/models/superdrive_data_masking_front/fuse_bn-checkpoint-last.pth.tar",  # noqa
                allow_miss=True,
                verbose=True,
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=data_loaders,
    batch_processor=dict(
        type="MultiBatchProcessor",
        need_grad_update=False,
    ),
    device=None,
    callbacks=callbacks,
    log_interval=log_freq,
    share_callbacks=False,
)
