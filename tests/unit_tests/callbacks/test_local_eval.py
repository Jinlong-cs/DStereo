import os
import shutil

import cv2
import numpy as np
import pytest
import torch

from hat.registry import build_from_registry

try:
    import hatbc
except ImportError:
    hatbc = None

EVAL_DIR = "/horizon-bucket/auto_eval/adas_eval/eval_platform"
EVAL_DIR_AVAIABLE = os.path.exists(EVAL_DIR)
tmpdir = "./tmp_localeval_test"


def build_callback(
    tmpdir,
    task_name,
    eval_handler_kwargs,
    reformat_output_fn=None,
):
    cfg = dict(
        type="LocalEval",
        output_root=os.path.join(tmpdir, "example-pred-outs/"),
        reformat_output_fn=reformat_output_fn,
        reformat_out_fn_kwargs=dict(),
        task_name=task_name,
        eval_handler_kwargs=eval_handler_kwargs,
    )
    callback = build_from_registry(cfg)
    return callback


@pytest.mark.skipif(hatbc is None, reason="need habtc")
@pytest.mark.skipif(not EVAL_DIR_AVAIABLE, reason="requiring auto_eval bucket")
def test_local_eval_det2d():
    def reformat_fn_det2d(batch, model_outs, **kwargs):
        results = []
        model_outs = model_outs["bboxes"]

        assert len(batch) == len(model_outs)

        for batch_i, out_i in zip(batch, model_outs):
            ret = {}
            ret["image_key"] = str(batch_i["image_key"])
            predicts = []
            for out_ij in out_i:
                if isinstance(out_ij, torch.Tensor):
                    out_ij = out_ij.cpu().numpy().tolist()
                data_ij = {"bbox": out_ij[:4], "bbox_score": out_ij[-1]}
                predicts.append(data_ij)

            ret["vehicle"] = predicts
            results.append(ret)

        return results

    setting = f"{EVAL_DIR}/fs/6028397/settings/7342430/urban_night_0to50_merge_0.5_8m.yaml"  # noqa
    gt = f"{EVAL_DIR}/fs/6028397/datasets/CN0820_220304/data.json"
    callback = build_callback(
        tmpdir,
        task_name="Detection2D",
        eval_handler_kwargs={"setting_file": setting, "gt_file": gt},
        reformat_output_fn=reformat_fn_det2d,
    )
    batch = [
        {
            "image_key": "night/urban/629316_2/data/ADAS_20211201-171522_800_1__276630_1638350261554_0.jpg",  # noqa
        },
    ]

    model_outs = {
        "bboxes": [
            torch.from_numpy(np.array([[215.0, 1031.0, 348.0, 1426.0, 0.99]])),
        ]
    }

    callback.on_loop_begin()
    callback.on_batch_end(batch=batch, model_outs=model_outs)
    callback.on_loop_end()
    shutil.rmtree(tmpdir)


@pytest.mark.skipif(not EVAL_DIR_AVAIABLE, reason="requiring auto_eval bucket")
def test_local_eval_seg():
    def reformat_fn_seg(batch, model_outs, **kwargs):
        results = []
        model_outs = model_outs["iqa_parsing"]

        assert len(batch) == len(model_outs)
        if isinstance(model_outs, torch.Tensor):
            model_outs = model_outs.cpu().numpy()
        model_outs = model_outs.astype("float32")
        for idx, batch_i in enumerate(batch):
            img_name = batch_i["img_key"]
            out_img = model_outs[idx]
            out_json = {"out_img": out_img, "image_name": img_name}
            results.append(out_json)
        return results

    images_dir = f"{EVAL_DIR}/fs/6042798/datasets/data/images"
    gt_dir = f"{EVAL_DIR}/fs/6042798/datasets/data/labels"
    config_file = f"{EVAL_DIR}/fs/6031633/settings/7361205/seg-image-fail_12cam_roi-all_cls7_mgt50.yaml"  # noqa
    callback = build_callback(
        tmpdir,
        task_name="Segmentaion",
        eval_handler_kwargs=dict(
            images_dir=images_dir,
            gt_dir=gt_dir,
            config_file=config_file,
            images_json=None,
            attr_path=None,
        ),
        reformat_output_fn=reformat_fn_seg,
    )
    batch = []
    for i in os.listdir(images_dir):
        batch.append({"img_key": i})
    pred = []
    for i in os.listdir(gt_dir):
        pred.append(cv2.imread(os.path.join(gt_dir, i), -1)[np.newaxis])
    pred = np.concatenate(pred)
    model_outs = {
        "iqa_parsing": torch.from_numpy(pred),
    }
    callback.on_loop_begin()
    callback.on_batch_end(batch=batch, model_outs=model_outs)
    callback.on_loop_end()
    shutil.rmtree(tmpdir)
