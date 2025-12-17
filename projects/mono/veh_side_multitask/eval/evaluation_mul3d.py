import argparse
import datetime
import json
import os
from glob import glob

import cv2
import horizon_plugin_pytorch
import numpy as np
import torch
import torch.utils.data as data
from evaluation_client import EvaluationClient
from tqdm import tqdm

from hat.data.transforms.real3d import ImageTransform
from hat.registry import build_from_registry
from hat.utils.apply_func import to_cuda
from hat.utils.checkpoint import load_checkpoint
from hat.utils.config import Config
from hat.visualize.real3d import (
    compute_box_3d,
    draw_projected_box3d,
    project_to_image,
)

LOCAL_TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class Real3dEvalDataset(data.Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.annos = f.readlines()
        self.json_path = json_path
        self.dirname = os.path.dirname(json_path)
        self.imagetransform = ImageTransform(
            size=(960, 192), center_shift=(0, 184.0)
        )

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        anno = json.loads(self.annos[idx])
        img_name = os.path.join(
            self.json_path[:-5], anno.get("image_key", anno.get("img_key"))
        )
        camera_default = (
            anno.get("img_anno", {})
            .get("attrs", {})
            .get("camera_default", None)
        )
        if camera_default:
            focal_u = camera_default.get("focal_u")
            focal_v = camera_default.get("focal_v")
            center_u = camera_default.get("center_u")
            center_v = camera_default.get("center_v")
            calib = np.array(
                [
                    [focal_u, 0, center_u, 0],
                    [0, focal_v, center_v, 0],
                    [0, 0, 1, 0],
                ]
            )
            calib = torch.as_tensor(calib, dtype=torch.float32)
            distCoeffs = np.array(
                camera_default["dist_coeffs"], dtype=np.float32
            )
            distCoeffs = torch.as_tensor(distCoeffs, dtype=torch.float32)
        else:
            calib = np.array(anno["calib"], dtype=np.float32)
            distCoeffs = np.array(anno["distCoeffs"], dtype=np.float32)
        image = cv2.imread(img_name)
        data = self.imagetransform({"img": image})
        inputs = {
            "img": data["img"],
            "orig_img": image,
            "img_name": img_name,
            "calibration": calib,
            "dist_coeffs": distCoeffs,
            "image_transform": data["image_transform"],
        }
        return inputs


def vehicle_3d_eval(
    model_name,
    model,
    dataset_id=None,
    device=None,
    cfg_file=None,
    vis_result=False,
    data_path=None,
):
    dataset_id = str(dataset_id)
    json_file = (
        data_path
        if data_path
        else glob(
            os.path.join(
                f"/horizon-bucket/auto_eval/adas_eval/eval_platform/fs/{dataset_id}/datasets/*",
                "data.json",
            )
        )[0]
    )
    prediction_file_path = os.path.join(
        "tmp_eval",
        model_name,
        "real3d",
        LOCAL_TIME,
        os.path.basename(data_path) if data_path else dataset_id,
    )
    if not prediction_file_path.endswith(".json"):
        prediction_file_path += ".json"
    os.makedirs(os.path.dirname(prediction_file_path), exist_ok=True)
    vis_save_path = prediction_file_path[:-5]
    os.makedirs(vis_save_path, exist_ok=True)
    dataset = Real3dEvalDataset(json_file)
    dataloader = data.DataLoader(
        dataset,
        batch_size=2 if os.getenv("TEST_MODE", "0") == "1" else 64,
        num_workers=2 if os.getenv("TEST_MODE", "0") == "1" else 8,
        shuffle=False,
    )

    pred_json_lines = []
    with torch.no_grad():
        for batch in tqdm(dataloader):

            # batch["img_id"] = list(range(128))
            # batch["layout"] = ["chw" * 128]
            # batch["pad_shape"] = torch.Tensor([[3, 192, 960]]).expand(128,3)

            batch = to_cuda(batch, device)
            # patch for 2d
            inputs_2d = dict(
                img_id=None,
                # img_name=None,
                img_height=None,
                img_width=None,
                color_space=None,
                layout=None,
                scale_factor=None,
                img_shape=None,
                resized_shape=None,
                pad_shape=None,
                keep_ratio=None,
                # unneccesay key-value
                scale=None,
                scale_idx=None,
                # TODO(min.du): using patch #
                crop_offset=None,
                before_crop_shape=None,
                # crop_bbox=None,
                padded_img=None,
                before_pad_shape=None,
            )
            batch.update(inputs_2d)

            batch[
                "img"
            ] = horizon_plugin_pytorch.nn.bgr_to_yuv444.bgr_to_yuv444(
                batch["img"].permute(0, 3, 1, 2), False
            )
            batch["img"] = (batch["img"].float() - 128.0) / 128.0
            model_outs = model(batch)[-1]
            predict_dim = (
                model_outs.real3d_real3d_head_predict_dim.cpu().numpy()
            )
            predict_score = (
                model_outs.real3d_real3d_head_predict_score.cpu().numpy()
            )
            predict_dep = (
                model_outs.real3d_real3d_head_predict_dep.cpu().numpy()
            )
            predict_location = (
                model_outs.real3d_real3d_head_predict_location.cpu().numpy()
            )
            predict_rotation_y = (
                model_outs.real3d_real3d_head_predict_rotation_y.cpu().numpy()
            )
            nms_keep = (
                model_outs.real3d_real3d_head_predict_nms_keep.cpu().numpy()
            )
            img_names = batch["img_name"]
            orig_imgs = batch["orig_img"]
            calibration = batch["calibration"].cpu()
            dist_coeffs = batch["dist_coeffs"].cpu()
            batch_size = predict_dim.shape[0]
            topk = predict_dim.shape[1]

            for b in range(batch_size):
                vehicle_json = {}
                vehicle_json["image_key"] = os.path.basename(
                    img_names[b]
                ).replace(".jpg", "")
                vehicle_json["vehicle"] = []
                if vis_result:
                    image = np.array(orig_imgs[b].cpu())
                for t in range(topk):
                    if nms_keep[b, t] and predict_score[b, t] > 0.3:
                        corners3d = compute_box_3d(
                            predict_dim[b, t],
                            predict_location[b, t],
                            predict_rotation_y[b, t],
                        )
                        corners3d_proj = project_to_image(
                            (3840, 2160),
                            corners3d,
                            calibration[b, :, :3],
                            dist_coeff=dist_coeffs[b],
                            fisheye=False,
                        )
                        corners3d_proj = corners3d_proj.reshape(-1, 2).astype(
                            np.int32
                        )
                        bbox2d = np.concatenate(
                            [
                                np.min(corners3d_proj, axis=0),
                                np.max(corners3d_proj, axis=0),
                            ]
                        )
                        bbox2d = np.round(bbox2d).astype(np.int32)
                        if vis_result:
                            image = draw_projected_box3d(
                                image, corners3d_proj, (0, 0, 255)
                            )
                        new_det = {}
                        new_det["score"] = predict_score[b, t].tolist()
                        new_det["depth"] = predict_dep[b, t].tolist()
                        new_det["dimensions"] = predict_dim[b, t].tolist()
                        new_det["rotation_y"] = predict_rotation_y[
                            b, t
                        ].tolist()
                        new_det["location"] = predict_location[b, t].tolist()
                        new_det["bbox_2d"] = bbox2d.tolist()
                        vehicle_json["vehicle"].append(new_det)
                pred_json_lines.append(json.dumps(vehicle_json) + "\n")
                if vis_result:
                    img_path = os.path.join(
                        vis_save_path, vehicle_json["image_key"] + ".jpg"
                    )
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    cv2.imwrite(img_path, image)
            if os.getenv("TEST_MODE", "0") == "1":
                break
    with open(prediction_file_path, "w") as f:
        f.writelines(pred_json_lines)
    return prediction_file_path


def submit_evaluation(
    token, dataset_id, prediction_name, prediction_file_path, tags
):
    dataset_id = str(dataset_id)
    if os.getenv("TEST_MODE", "0") == "1":
        print(
            f"Skip submitting eval res {prediction_file_path} to {dataset_id} for pipeline-test mode."
        )
        return
    host = "http://model.aidi.hobot.cc"
    # project_id = "PDT2020005"
    project_id = "PDT20220004"
    client = EvaluationClient(host)
    client.auth.login_by_ldap(token)
    pr = (
        client.prediction.set_project_id(project_id)
        .set_queue_name("svc-aip-cpu")
        .set_running_resource(cpu=2, cpu_mem_ratio=4)
    )
    pr.create(
        dataset_id=dataset_id,
        prediction_name=prediction_name,
        prediction_file_path=prediction_file_path,
        tags=tags,
        remarks="this is example prediction",
    )
    # os.remove(prediction_file_path)


def vehicle_side_eval(
    model_name,
    model,
    dataset_id,
    device,
    cfg_file,
    vis_result=True,
    batch_patch=None,
    data_path=None,
    thread_score=0.5,
):
    # find eval dataset img dir
    dataset_id = str(dataset_id)
    dirname = f"/horizon-bucket/auto_eval/adas_eval/eval_platform/fs/{dataset_id}/datasets"  # noqa
    cfg = Config.fromfile(cfg_file.replace("multitask.py", "vehicle_side.py"))
    val_data_loader_cfg = cfg["test_data_loader"]
    val_data_loader_cfg["dataset"]["data_path"] = (
        data_path if data_path else dirname
    )
    val_data_loader_cfg["dataset"]["return_orig_img"] = vis_result
    val_data_loader_cfg["batch_size"] = (
        2 if os.getenv("TEST_MODE", "0") == "1" else 64
    )
    val_data_loader_cfg["num_workers"] = (
        2 if os.getenv("TEST_MODE", "0") == "1" else 8
    )

    dataloader = build_from_registry(val_data_loader_cfg)

    prediction_file_path = os.path.join(
        "tmp_eval",
        model_name,
        "vehicleside",
        LOCAL_TIME,
        os.path.basename(data_path) if data_path else dataset_id,
    )
    if not prediction_file_path.endswith(".json"):
        prediction_file_path += ".json"
    os.makedirs(os.path.dirname(prediction_file_path), exist_ok=True)
    vis_save_path = prediction_file_path[:-5]
    os.makedirs(vis_save_path, exist_ok=True)
    pred_json_lines = []

    with torch.no_grad():
        for batch in tqdm(dataloader):

            img_names = batch["img_name"]
            if not vis_result:
                batch.update({"orig_img": [0]})
            imgs: torch.Tensor = batch["orig_img"] if vis_result else None
            batch = to_cuda(batch, device)
            if batch_patch:
                batch.update(batch_patch)

            model_outs = model(batch)[0]
            polygons_list = (
                model_outs.vehicle_side_vehicle_side_head_predict_pred_polygons
            )
            for i in range(len(img_names)):
                img_name = img_names[i]
                if vis_result:
                    img = imgs[i]
                    if isinstance(img, torch.Tensor):
                        img: np.ndarray = img.squeeze().cpu().numpy()
                    assert img.ndim == 3
                polygons = polygons_list[i]
                if isinstance(polygons, torch.Tensor):
                    polygons = polygons.cpu().numpy()
                anno = dict(image_key=img_name, vehicle_side=[])
                for polygon_id in range(len(polygons)):
                    polygon = polygons[polygon_id]
                    if vis_result:
                        score = float(polygon[-2])
                        if score > thread_score:
                            pts = np.array(
                                [
                                    list(polygon[ii : ii + 2])
                                    for ii in range(0, 8, 2)
                                ],
                                dtype=np.int32,
                            )
                            pts = pts.reshape((-1, 1, 2))
                            cv2.polylines(img, [pts], 1, (0, 0, 255), 2)
                            cv2.putText(
                                img,
                                f"{score:.2f}",
                                (int(polygon[0]), int(polygon[1])),
                                cv2.FONT_HERSHEY_PLAIN,
                                2.5,
                                (0, 0, 255),
                                7,
                            )
                    bbox = [
                        polygon[0],
                        polygon[1],
                        polygon[4],
                        (polygon[5] + polygon[7]) / 2,
                    ]
                    bbox_dc = dict(
                        id=int(polygon_id),
                        bbox=list(map(float, bbox)),
                        polygon=list(map(float, polygon)),
                        bbox_score=float(polygon[-2]),
                        attrs={},
                    )
                    anno["vehicle_side"].append(bbox_dc)
                anno_str = json.dumps(anno)
                pred_json_lines.append(anno_str + "\n")
                if vis_result:
                    img_path = os.path.join(vis_save_path, img_name)
                    # bgr --> rgb
                    # img = img[..., [2, 1, 0]]
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    cv2.imwrite(img_path, img)
            if os.getenv("TEST_MODE", "0") == "1":
                break
    with open(prediction_file_path, "w") as f:
        f.writelines(pred_json_lines)
    return prediction_file_path


def fcos_2d_eval(
    model_name,
    model,
    dataset_id,
    device,
    cfg_file,
    eval_cls_name,
    eval_cls_id,
    model_output_id=1,
    vis_result=False,
    batch_patch=None,
    data_path=None,
    thread_score=0.5,
):
    dataset_id = str(dataset_id)
    dirname = f"/horizon-bucket/auto_eval/adas_eval/eval_platform/fs/{dataset_id}/datasets"  # noqa
    cfg_file = cfg_file.replace("multitask.py", "face_plate.py")
    cfg = Config.fromfile(cfg_file)
    val_data_loader_cfg = cfg["test_data_loader"]
    val_data_loader_cfg["dataset"]["data_path"] = (
        data_path if data_path else dirname
    )
    val_data_loader_cfg["dataset"]["return_orig_img"] = vis_result
    val_data_loader_cfg["batch_size"] = (
        2 if os.getenv("TEST_MODE", "0") == "1" else 64
    )
    val_data_loader_cfg["num_workers"] = (
        2 if os.getenv("TEST_MODE", "0") == "1" else 8
    )

    dataloader = build_from_registry(val_data_loader_cfg)

    prediction_file_path = os.path.join(
        "tmp_eval",
        model_name,
        eval_cls_name,
        LOCAL_TIME,
        os.path.basename(data_path) if data_path else dataset_id,
    )
    if not prediction_file_path.endswith(".json"):
        prediction_file_path += ".json"
    os.makedirs(os.path.dirname(prediction_file_path), exist_ok=True)
    vis_save_path = prediction_file_path[:-5]
    os.makedirs(vis_save_path, exist_ok=True)
    pred_json_lines = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            img_names = batch["img_name"]
            if not vis_result:
                batch.update({"orig_img": [0]})
            imgs: torch.Tensor = batch["orig_img"] if vis_result else None
            batch = to_cuda(batch, device)
            # patch for real3d
            if batch_patch:
                batch.update(batch_patch)

            model_outs = model(batch)

            model_outs: dict = model_outs[model_output_id]._asdict()

            for k, v in model_outs.items():
                if k.endswith("pred_bboxes"):
                    pred_bboxes = v
                    break
            for i in range(len(img_names)):
                img_name = img_names[i]

                pred_bbox = pred_bboxes[i]
                if isinstance(pred_bbox, torch.Tensor):
                    pred_bbox = pred_bbox.cpu().numpy()
                if vis_result:
                    img = imgs[i]
                    if isinstance(img, torch.Tensor):
                        img: np.ndarray = img.squeeze().cpu().numpy()
                    assert img.ndim == 3
                anno = dict(image_key=img_name)
                anno[eval_cls_name] = []
                for bbox in pred_bbox:
                    cls_id = int(bbox[5])
                    if cls_id != eval_cls_id:
                        continue
                    if vis_result:
                        score = float(bbox[4])
                        if score > thread_score:
                            x1, y1, x2, y2 = np.rint(bbox[:4]).astype(np.int32)
                            if cls_id == 0:
                                cv2.rectangle(
                                    img, (x1, y1), (x2, y2), (255, 0, 0), 1
                                )
                                print(img_name)
                            else:
                                cv2.rectangle(
                                    img, (x1, y1), (x2, y2), (0, 0, 255), 1
                                )
                            cv2.putText(
                                img,
                                f"{score:.2f}",
                                (int(x1), int(y1)),
                                cv2.FONT_HERSHEY_PLAIN,
                                2.5,
                                (0, 0, 255),
                                7,
                            )

                    bbox_dc = dict(
                        bbox=bbox[:4].tolist(),
                        bbox_score=float(bbox[4]),
                        attrs={},
                    )
                    anno[eval_cls_name].append(bbox_dc)
                anno_str = json.dumps(anno)
                pred_json_lines.append(anno_str + "\n")
                if vis_result:
                    # img_name = os.path.basename(img_name)
                    img_path = os.path.join(vis_save_path, img_name)
                    # bgr --> rgb
                    # img = img[..., [2, 1, 0]]
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    cv2.imwrite(img_path, img)
            if os.getenv("TEST_MODE", "0") == "1":
                break
    with open(prediction_file_path, "w") as f:
        f.writelines(pred_json_lines)
    return prediction_file_path


def run_one(
    model_name,
    model_version,
    stage,
    pretrained_ckpt=None,
    merge_face_plate=True,
    tasks=None,
    prediction_name=None,
    vis_result=True,
    data_paths: str = None,
    pipeline_test: bool = False,
):
    if stage == "int_inference":
        ckpt_stage = "qat"
    elif stage == "IntInference":
        ckpt_stage = "Qat"
    else:
        ckpt_stage = stage
    if pipeline_test:
        os.environ["TEST_MODE"] = "1"
    if tasks is None:
        tasks = "real3d vehicleside face plate"
    cfg_file = f"projects/mono/{model_name}/{model_version}/multitask.py"
    if data_paths:
        data_paths = data_paths.split(",")
    device = torch.device("cuda:0")
    cfg = Config.fromfile(cfg_file)
    model = cfg["test_model"]
    model = build_from_registry(model)
    os.system(f"rm -rf {os.getenv('HOME')}/.cache/torch/hub/checkpoints")
    horizon_plugin_pytorch.march.set_march(cfg.march)
    if pretrained_ckpt is None:
        from aidisdk import AIDIClient

        tmp_dir = f"~/tmp_aidi_models/{model_name}_{model_version}"
        os.system(f"mkdir -p {tmp_dir}")
        client = AIDIClient()
        pretrained_ckpt = client.model.download(
            tmp_dir, model_name, model_version, ckpt_stage
        )
    checkpoint = load_checkpoint(pretrained_ckpt, check_hash=False)
    if stage in [
        "float",
        "freezebn",
        "freeze_bn",
        "float_freeze_bn",
        "Float",
        "FreezeBn",
        "FloatFreezeBn",
    ]:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    elif stage in ["qat", "Qat"]:
        model.fuse_model()
        model.set_qconfig()
        model = horizon_plugin_pytorch.quantization.prepare_qat(
            model, inplace=False
        )
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    elif stage in ["int_inference", "IntInference"]:
        model.fuse_model()
        model.set_qconfig()
        model = horizon_plugin_pytorch.quantization.prepare_qat(
            model, inplace=False
        )
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        horizon_plugin_pytorch.quantization.convert(model.eval(), inplace=True)
        # model = horizon_plugin_pytorch.quantization.convert(
        #     model, inplace=False
        # )
    model = model.eval().to(device)

    if prediction_name is None or prediction_name == "":
        prediction_name = f"{model_name}_{model_version}_{stage}"
    print("prediction_name:", prediction_name)
    tags = [
        "vehicle_side",
        "det",
        "hat",
        stage,
    ]
    token = None
    batch_patch = None
    if "real3d_head" in model.nodes:
        real3d_decoder = model.nodes["real3d_head"].postprocess
        model.nodes["real3d_head"].postprocess = None
        batch_patch = dict(
            calibration=None,
            dist_coeffs=None,
            image_transform=None,
        )

    if "vehicleside" in tasks:
        dataset_ids = [
            # normal
            "6040553",
            # cross
            "6040601",
            # special
            "6041275",
            # csc
            "6042266",
        ]
        if data_paths:
            for data_path in data_paths:
                prediction_file_path = vehicle_side_eval(
                    prediction_name,
                    model,
                    None,
                    device,
                    cfg_file,
                    vis_result=vis_result,
                    batch_patch=batch_patch,
                    data_path=data_path,
                )
        else:
            for dataset_id in dataset_ids:
                prediction_file_path = vehicle_side_eval(
                    prediction_name,
                    model,
                    dataset_id,
                    device,
                    cfg_file,
                    vis_result=vis_result,
                    batch_patch=batch_patch,
                )
                submit_evaluation(
                    token,
                    dataset_id,
                    prediction_name,
                    prediction_file_path,
                    tags,
                )
    if "face" in tasks:
        if data_paths:
            for data_path in data_paths:
                prediction_file_path = fcos_2d_eval(
                    prediction_name,
                    model,
                    None,
                    device,
                    cfg_file,
                    "face",
                    0,
                    1,
                    vis_result=vis_result,
                    batch_patch=batch_patch,
                    data_path=data_path,
                )
        else:
            dataset_ids = [
                "6041137",
                "6040762",
            ]
            for dataset_id in dataset_ids:
                prediction_file_path = fcos_2d_eval(
                    prediction_name,
                    model,
                    dataset_id,
                    device,
                    cfg_file,
                    "face",
                    0,
                    1,
                    vis_result=vis_result,
                    batch_patch=batch_patch,
                )
                submit_evaluation(
                    token,
                    dataset_id,
                    prediction_name,
                    prediction_file_path,
                    tags,
                )
    if "plate" in tasks:
        if data_paths:
            for data_path in data_paths:
                prediction_file_path = fcos_2d_eval(
                    prediction_name,
                    model,
                    None,
                    device,
                    cfg_file,
                    "plate",
                    1 if merge_face_plate else 0,
                    1 if merge_face_plate else 2,
                    vis_result=vis_result,
                    batch_patch=batch_patch,
                    data_path=data_path,
                )
        else:
            dataset_ids = [
                "6041153",
                "6040434",
            ]
            for dataset_id in dataset_ids:
                prediction_file_path = fcos_2d_eval(
                    prediction_name,
                    model,
                    dataset_id,
                    device,
                    cfg_file,
                    "plate",
                    1 if merge_face_plate else 0,
                    1 if merge_face_plate else 2,
                    vis_result=vis_result,
                    batch_patch=batch_patch,
                )
                submit_evaluation(
                    token,
                    dataset_id,
                    prediction_name,
                    prediction_file_path,
                    tags,
                )

    if "real3d" in tasks:
        model.nodes["real3d_head"].postprocess = real3d_decoder
        model.nodes["vehicle_side_head"].postprocess = None
        if merge_face_plate:
            model.nodes["face_plate_head"].postprocess = None
        else:
            model.nodes["face_head"].postprocess = None
            model.nodes["plate_head"].postprocess = None

        dataset_ids = [
            # 6041846,  # J2评测集
            "6040713",  # 非同源
            "6036724",  # GL106 cross
            # 6031891,  # CA110
            # "6035976",  # GL106 normal
            # 6038635,  # GL888 normal
            # 6038634,  # GL888 cross
            # 6030033,  # JL normal
            # 6028529,  # LX 0820
        ]
        if data_paths:
            for data_path in data_paths:
                prediction_file_path = vehicle_3d_eval(
                    prediction_name,
                    model,
                    None,
                    device,
                    cfg_file,
                    vis_result=vis_result,
                    data_path=data_path,
                )
        else:
            for dataset_id in dataset_ids:
                prediction_file_path = vehicle_3d_eval(
                    prediction_name,
                    model,
                    dataset_id,
                    device,
                    cfg_file,
                    vis_result=vis_result,
                )
                submit_evaluation(
                    token,
                    dataset_id,
                    prediction_name,
                    prediction_file_path,
                    tags,
                )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-version", type=str, required=True)
    parser.add_argument("--stage", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--merge-faceplate", action="store_true")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        choices=["real3d", "face", "plate", "vehicleside"],
    )
    parser.add_argument("--prediction-name", type=str, default=None)
    parser.add_argument("--data-paths", type=str, default=None)
    parser.add_argument("--vis-result", action="store_true")
    parser.add_argument("--pipeline-test", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    run_one(
        args.model_name,
        args.model_version,
        args.stage,
        args.ckpt,
        args.merge_faceplate,
        "".join(args.tasks),
        args.prediction_name,
        args.vis_result,
        args.data_paths,
        args.pipeline_test,
    )
