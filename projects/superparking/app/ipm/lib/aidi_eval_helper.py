import json
import os

from torch.nn import functional as F

CONFIG_DIR = os.path.dirname(__file__)
is_local = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local else "/bucket/input"

TASK2AIDI_EVAL_IDS = {
    "psd": [6036149],
    "parsing": [6039118],
}

aidi_val_datapath = os.path.join(
    bucket_root, "auto_eval/adas_eval/eval_platform/fs/"
)
TASK2AIDI_EVAL_DATA_PATH = {
    task_name: [
        os.path.join(aidi_val_datapath, str(id), "datasets")
        for id in dataset_id
    ]
    for task_name, dataset_id in TASK2AIDI_EVAL_IDS.items()
}


def reformat_psd_to_aidi_eval(
    batch_data,
    batch_outputs,
):
    pred_psd = json.loads(batch_outputs[0][0])["preds_slots"]
    image_names = batch_data[0]["img_name"]
    batch_results = []
    for image_i, image_name in enumerate(image_names):
        batch_results.append(
            {
                "image_key": image_name,
                "pred_slots": [pred_psd[image_i]],
            }
        )

    return batch_results


def reformat_seg_to_aidi_eval(
    batch_data,
    batch_outputs,
):
    if isinstance(batch_data, tuple):
        assert isinstance(
            batch_data[1], str
        ), f"Batch format error, it should be (batch, object_name), \
             but {batch_data}"
        batch_data = batch_data[0]
    batch_objects = batch_outputs[0][0]
    rets = []
    ori_hws = batch_data["orig_hw"]
    for _, (img_name, objects, h_i, w_i) in enumerate(
        zip(batch_data["img_name"], batch_objects, ori_hws[0], ori_hws[1])
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
