import json
import os
import random
import time

import cv2
import pytest
import torch
from torch.utils.data._utils.collate import default_convert

from hat.callbacks.aidi_eval import AIDIEval, AIDIReport
from hat.data.datasets.rand_dataset import RandDataset
from hat.engine import Predictor
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.engine.processors.processor import MultiBatchProcessor
from hat.registry import build_from_registry
from hat.utils.aidi import get_aidi_token
from hat.utils.apply_func import convert_numpy
from hat.utils.filesystem import file_load
from tests.data.toy_modules import ToyBackbone, ToyHead, ToyLoss, ToyModel

try:
    import aidisdk
except ImportError:
    aidisdk = None


cameras = [
    "camera_front",
    "camera_front_left",
    "camera_front_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
]


def reformat_fn_det2d(batch, model_outs, **kwargs):
    results = []
    model_outs = model_outs["bboxes"]

    assert len(batch) == len(model_outs)

    for batch_i, out_i in zip(batch, model_outs):
        ret = {}
        ret["image_key"] = str(batch_i["img_key"])
        predicts = []
        for out_ij in out_i:
            if isinstance(out_ij, torch.Tensor):
                out_ij = out_ij.cpu().numpy().tolist()
            data_ij = {"bbox": out_ij[:4], "score": out_ij[-1]}
            predicts.append(data_ij)

        ret["person"] = predicts
        results.append(ret)

    return results


def reformat_fn_seg(batch, model_outs, **kwargs):
    results = []
    model_outs = model_outs["lane_lane_head_predict_pred_seg"]

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


def reformat_fn_real3d(batch, model_outs, **kwargs):
    results = {}
    results["batch"] = batch
    results["outputs"] = model_outs
    results = convert_numpy(results)
    return results


def reformat_fn_bevseg(batch, model_outs, **kwargs):
    img_name = []
    timetamp = batch["timestamp"]
    timetamp = convert_numpy(timetamp)
    timetamp = timetamp.tolist()
    for imgname in timetamp:
        imgname = str(int(imgname[0])) + ".png"
        img_name.append(imgname)
    bev_batch = {}
    bev_outputs = {}
    bev_outputs["img_name"] = img_name

    results = {}
    results["batch"] = bev_batch
    results["outputs"] = bev_outputs
    results["predict_results"] = model_outs["bev_seg_head_predict"][0]
    results = convert_numpy(results)

    return results


def reformat_fn_bev3d(batch, model_outs, **kwargs):
    model_outs = convert_numpy(model_outs)
    pred_output = {}
    pred_keys = [
        "bev3d_ct",
        "bev3d_cls_id",
        "bev3d_score",
        "bev3d_rot",
        "bev3d_dim",
        "bev3d_loc_z",
    ]
    for k, v in model_outs.items():
        for key in pred_keys:
            if "predict_" + key in k:
                pred_output[key] = v

    timetamp = batch["timestamp"]
    timestamp = convert_numpy(timetamp.cpu()) * 1000
    timestamps = [str(int(_bs_time)) for _bs_time in timestamp]
    batch_size = pred_output["bev3d_ct"].shape[0]

    pred_list_side = []
    pred_list_front = []
    for idx in range(batch_size):
        front_img_timestamp = timestamps[idx]
        pack_dir = batch["pack_dir"][idx]
        key = os.path.join(pack_dir, front_img_timestamp)
        for _, cam in enumerate(cameras):
            coco_image = {
                "image_key": key.split("/")[-2]
                + "__"
                + cam
                + "__"
                + "1646980552800.jpg",
                "cyclist": [],
            }
            ann = {
                "bbox_2d": torch.rand([4]),
                "depth": torch.rand([1]),
                "dimensions": torch.rand([3]),
                "rotation_y": torch.rand([1]),
                "location": torch.rand([3]),
                "score": torch.rand([1]),
            }
            coco_image["cyclist"].append(ann)

            if cam == "camera_front":
                pred_list_front.append(coco_image)
            else:
                pred_list_side.append(coco_image)
    results = {}
    results["front_outputs"] = pred_list_front
    results["side_outputs"] = pred_list_side

    return results


def build_callback(
    tmpdir,
    aidi_eval_dataset_id=None,
    aidi_eval_dataset_name=None,
    reformat_output_fn=None,
):
    cfg = dict(
        type="AIDIEval",
        output_root=os.path.join(tmpdir, "example-pred-outs"),
        project_id="RDS20220011",
        prediction_name="example-prediction-1234",
        prediction_tags="example",
        aidi_eval_dataset_id=aidi_eval_dataset_id,
        aidi_eval_dataset_name=aidi_eval_dataset_name,
        reformat_output_fn=reformat_output_fn,
        reformat_out_fn_kwargs=dict(),
    )
    callback = build_from_registry(cfg)
    return callback


@pytest.mark.skipif(aidisdk is None, reason="need aidisdk")
class TestAidieval(object):
    def test_aidi_eval_det2d(self, tmpdir):
        callback = build_callback(
            tmpdir,
            aidi_eval_dataset_id=[6027150],
            reformat_output_fn=reformat_fn_det2d,
        )

        batch = [
            {"img_key": "hat_0001.png", "data": torch.randn(3, 16, 16)},
            {"img_key": "hat_0002.png", "data": torch.randn(3, 16, 16)},
        ]

        model_outs = {"bboxes": [torch.randn([2, 5]), torch.randn([3, 5])]}
        callback.on_loop_begin()
        callback.on_batch_end(batch=batch, model_outs=model_outs)
        save_dir = callback.output_root
        json_file = os.path.join(save_dir, os.listdir(save_dir)[0])
        assert os.path.exists(json_file)

        with open(json_file, "r") as f:
            for idx, line in enumerate(f):
                preds_data = json.loads(line)
                assert preds_data["image_key"] == batch[idx]["img_key"]
                assert len(preds_data["person"]) == model_outs["bboxes"][
                    idx
                ].size(0)

    def test_aidi_eval_seg(self, tmpdir):
        callback = build_callback(
            tmpdir,
            aidi_eval_dataset_id=[6026266],
            reformat_output_fn=reformat_fn_seg,
        )

        batch = [
            {"img_key": "hat_0001.png", "data": torch.randn(3, 16, 16)},
            {"img_key": "hat_0002.png", "data": torch.randn(3, 16, 16)},
        ]
        model_outs = {
            "lane_lane_head_predict_pred_seg": torch.randn(2, 24, 24),
        }
        callback.on_loop_begin()
        callback.on_batch_end(batch=batch, model_outs=model_outs)
        save_dir = callback.output_root
        dir_name = os.path.join(save_dir, os.listdir(save_dir)[1])
        assert os.path.exists(dir_name)
        assert len(os.listdir(dir_name)) == len(batch)
        for idx, img_name in enumerate(sorted(os.listdir(dir_name))):
            assert img_name == batch[idx]["img_key"]
            img_path = os.path.join(dir_name, img_name)
            assert cv2.imread(img_path).shape == (24, 24, 3)

    def test_aidi_eval_real3d(self, tmpdir):
        callback = build_callback(
            tmpdir,
            aidi_eval_dataset_id=[6027631],
            reformat_output_fn=reformat_fn_real3d,
        )

        batch = {
            "image_name": ["real3d_001.jpg", "real3d_002.jpg"],
            "image_width": torch.tensor([3840, 3840]),
            "image_height": torch.tensor([2160, 2160]),
            "calibration": torch.randn([2, 3, 4]),
            "image_id": ["123", "456"],
            "dist_coeffs": torch.randn([2, 8]),
        }
        model_outs = {
            "real3d_fov120_real3d_head_predict_alpha": torch.randn([2, 40]),
            "real3d_fov120_real3d_head_predict_bbox": torch.randn([2, 40, 4]),
        }

        callback.on_loop_begin()
        callback.on_batch_end(batch=batch, model_outs=model_outs)
        save_dir = callback.output_root
        dir_name = os.path.join(save_dir, os.listdir(save_dir)[0])
        assert os.path.exists(dir_name)
        assert len(os.listdir(dir_name)) == 1
        pkl_file_path = os.path.join(dir_name, os.listdir(dir_name)[0])
        assert os.path.exists(pkl_file_path)

        for preds_data in file_load(pkl_file_path):
            preds_data = default_convert(preds_data)
            preds_batch = preds_data["batch"]
            preds_outputs = preds_data["outputs"]
            assert len(preds_batch["image_name"]) == len(batch["image_name"])
            assert preds_outputs[
                "real3d_fov120_real3d_head_predict_alpha"
            ].size(0) == model_outs[
                "real3d_fov120_real3d_head_predict_alpha"
            ].size(
                0
            )

    @pytest.mark.skip(
        reason="datasetid don't exists on the platform, skip this."
    )
    def test_aidi_eval_bevseg(self, tmpdir):
        callback = build_callback(
            tmpdir,
            aidi_eval_dataset_id=[6028522],
            reformat_output_fn=reformat_fn_bevseg,
        )

        batch = {"timestamp": torch.tensor([[12], [34], [56]])}
        model_outs = {"bev_seg_head_predict": [torch.randn([3, 36, 36])]}

        callback.on_loop_begin()
        callback.on_batch_end(batch=batch, model_outs=model_outs)
        save_dir = callback.output_root
        dir_name = os.path.join(save_dir, os.listdir(save_dir)[0])
        assert os.path.exists(dir_name)
        assert len(os.listdir(dir_name)) == 4

        for file_name in os.listdir(dir_name):
            if file_name.endswith(".pkl"):
                assert file_name == "result.pkl"
            else:
                assert file_name in [
                    "bev_seg_12.png",
                    "bev_seg_34.png",
                    "bev_seg_56.png",
                ]
                img_path = os.path.join(dir_name, file_name)
                assert cv2.imread(img_path).shape == (36, 36, 3)


@pytest.mark.skipif(aidisdk is None, reason="need aidisdk")
@pytest.mark.skip(reason="may produce unexpected results, skip this.")
def test_aidi_eval_in_loop(tmpdir):
    def fake_reformat_fn(batch, model_outs, **kwargs):
        fake_results = [{"image_key": "0001.png", "person": []}]
        return fake_results

    CI_TOKEN = get_aidi_token()
    predictor = Predictor(
        model=ToyModel(
            backbone=ToyBackbone(strides=(1, 2), channels=(3, 8)),
            head=ToyHead(
                in_channels=8,
                fc_filter=16,
                num_classes=10,
                with_dequant=True,
            ),
            loss=ToyLoss(),
        ),
        data_loader=torch.utils.data.DataLoader(
            dataset=RandDataset(
                length=4,
                example=(
                    torch.randn((3, 14, 14)),
                    random.randint(0, 10 - 1),
                ),  # noqa E501
                clone=True,
            ),
            batch_size=2,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        ),
        batch_processor=MultiBatchProcessor(
            need_grad_update=False,
            loss_collector=collect_loss_by_regex("^.*loss.*"),
        ),
        device=0,
        callbacks=[
            AIDIEval(
                output_root=os.path.join(tmpdir, "example-pred-outs/"),
                project_id="RDS20220011",
                prediction_name="example-prediction-1234",
                prediction_tags="example",
                aidi_eval_dataset_id=["6027150"],
                reformat_output_fn=fake_reformat_fn,
                reformat_out_fn_kwargs=dict(task_name="person"),
                aidi_eval_host="http://model.aidi.hobot.cc",
                aidi_eval_token=CI_TOKEN,
            )
        ],
    )
    try:
        predictor.fit()
    except Exception as e:
        # skip `create_prediction` error caused by CI_Token.
        if (
            str(e)
            == "can not get user (hat_team) info from the projectID (RDS20220011) white list"  # noqa E501
        ):
            pass
        else:
            raise RuntimeError(str(e))
    save_dir = predictor.callbacks[0].output_root
    json_file = os.path.join(save_dir, os.listdir(save_dir)[0])

    assert os.path.exists(json_file)

    with open(json_file, "r") as f:
        for _, line in enumerate(f):
            preds_data = json.loads(line)
            assert preds_data["image_key"] == "0001.png"
            assert len(preds_data["person"]) == 0


@pytest.mark.skipif(aidisdk is None, reason="need aidisdk")
def test_aidi_report():
    timeArray = time.localtime(int(time.time()))
    time_now = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    time_now = time_now.replace(" ", "").replace("-", "").replace(":", "")
    report = AIDIReport(
        project_id="RDS20220011",
        report_name=f"CI_test_{time_now}",
        page_label_list=["vehicle_rear"],
        type_list=["detection"],
        old_prediction_list=["old_leaderboard_name"],
        new_prediction_list=["new_leaderboard_name"],
        datasets_id_list=[6026824],
    )

    report.on_epoch_end()
