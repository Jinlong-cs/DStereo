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
from torchvision import transforms
from torchvision.ops import nms
from tqdm import tqdm

from hat.core.box3d_utils import compute_box_3d
from hat.core.utils_3d import project_to_image
from hat.data.transforms.detection import PresetCrop, Resize
from hat.data.transforms.real3d import ImageTransform
from hat.registry import build_from_registry
from hat.utils.apply_func import to_cuda
from hat.utils.checkpoint import load_checkpoint
from hat.utils.config import Config


class Real3dEvalDataset(data.Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.annos = f.readlines()
        self.dirname = os.path.dirname(json_path)
        self.imagetransform = ImageTransform(
            size=(960, 192), center_shift=(0, 184.0)
        )

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        anno = json.loads(self.annos[idx])
        img_name = os.path.join(self.dirname, "data", anno["image_key"])
        calib = np.array(anno["calib"], dtype=np.float32)
        distCoeffs = np.array(anno["distCoeffs"], dtype=np.float32)
        image = cv2.imread(img_name)
        data = self.imagetransform({"img": image})
        inputs = {
            "img": data["img"],
            "img_name": img_name,
            "calibration": torch.as_tensor(calib, dtype=torch.float32),
            "dist_coeffs": torch.as_tensor(distCoeffs, dtype=torch.float32),
            "image_transform": data["image_transform"],
        }
        return inputs


class AnonymizationEvalDataset(data.Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.annos = f.readlines()
        self.dirname = os.path.dirname(json_path)
        self.imagetransform = transforms.Compose(
            [
                Resize(img_scale=(540, 960), keep_ratio=False),
                PresetCrop(
                    crop_top=220, crop_bottom=128, crop_left=0, crop_right=0
                ),
            ]
        )

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        anno = json.loads(self.annos[idx])
        img_name = os.path.join(self.dirname, "data", anno["image_key"])
        image = cv2.imread(img_name)
        h, w, c = image.shape
        data = self.imagetransform({"img": image, "layout": "hwc"})
        inputs = {
            "img": data["img"],
            "img_name": img_name,
            "scale": np.array(
                [w / 960.0, h / 540.0, w / 960.0, h / 540.0], dtype=np.float32
            ),
            "offset": np.array([0, 220, 0, 220], dtype=np.float32),
        }
        return inputs


def submit_evaluation(
    token, dataset_id, prediction_name, prediction_file_path, tags
):
    host = "http://model.aidi.hobot.cc"
    project_id = "PDT2020005"
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
    os.remove(prediction_file_path)


def vehicle_evaluation(model, dataset_id, device):
    cfg = Config.fromfile(
        "projects/mono/real3d_multitask/vehicle_detection.py"
    )
    decoder = cfg["decoder"]
    decoder = build_from_registry(decoder)
    decoder = decoder.to(device)
    dirname = f"/horizon-bucket/auto_eval/adas_eval/eval_platform/fs/{dataset_id}/datasets/*"  # noqa
    json_file = glob(os.path.join(dirname, "data.json"))[0]
    local_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prediction_file_path = f"vehicle_{local_time}_.json"
    dataset = Real3dEvalDataset(json_file)
    dataloader = data.DataLoader(
        dataset, batch_size=32, num_workers=8, shuffle=True
    )

    with torch.no_grad():
        writer = open(prediction_file_path, "w")
        for batch in tqdm(dataloader):
            batch = to_cuda(batch, device)
            batch[
                "img"
            ] = horizon_plugin_pytorch.nn.bgr_to_yuv444.bgr_to_yuv444(
                batch["img"].permute(0, 3, 1, 2), False
            )
            batch["img"] = (batch["img"].float() - 128.0) / 128.0
            outputs = model(batch)
            vehicle_outputs = outputs[0]
            pred = dict(
                hm=vehicle_outputs.vehicle_detection_pred_hm,
                dep=vehicle_outputs.vehicle_detection_pred_dep,
                rot=vehicle_outputs.vehicle_detection_pred_rot,
                dim=vehicle_outputs.vehicle_detection_pred_dim,
                loc_offset=vehicle_outputs.vehicle_detection_pred_loc_offset,
                wh=vehicle_outputs.vehicle_detection_pred_wh,
            )
            outputs = decoder(pred, batch)
            predict_dim = outputs["dim"].cpu().numpy()
            predict_score = outputs["score"].cpu().numpy()
            predict_dep = outputs["dep"].cpu().numpy()
            predict_location = outputs["location"].cpu().numpy()
            predict_rotation_y = outputs["rotation_y"].cpu().numpy()
            nms_keep = outputs["nms_keep"].cpu().numpy()
            img_names = batch["img_name"]
            calibration = batch["calibration"].cpu().numpy()
            dist_coeffs = batch["dist_coeffs"].cpu().numpy()
            batch_size = predict_dim.shape[0]
            topk = predict_dim.shape[1]

            for b in range(batch_size):
                vehicle_json = {}
                vehicle_json["image_key"] = os.path.basename(img_names[b])
                vehicle_json["vehicle"] = []
                # image = cv2.imread(img_names[b])
                for t in range(topk):
                    if nms_keep[b, t] and predict_score[b, t] > 0.05:
                        corners3d = compute_box_3d(
                            predict_dim[b, t],
                            predict_location[b, t],
                            predict_rotation_y[b, t],
                        )
                        corners3d_proj = project_to_image(
                            corners3d,
                            calibration[b, :, :3],
                            dist_coeff=dist_coeffs[b],
                            # fisheye=False,
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
                # cv2.imwrite("111.jpg", image)
                vehicle_json = str(vehicle_json).replace("'", '"')
                writer.write(vehicle_json)
                writer.write("\n")
    return prediction_file_path


def face_evaluation(model, dataset_id, device):
    dirname = f"/horizon-bucket/auto_eval/adas_eval/eval_platform/fs/{dataset_id}/datasets/*"  # noqa
    json_file = glob(os.path.join(dirname, "data.json"))[0]
    local_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prediction_file_path = f"face_{local_time}_.json"
    dataset = AnonymizationEvalDataset(json_file)
    dataloader = data.DataLoader(
        dataset, batch_size=32, num_workers=8, shuffle=True
    )

    with torch.no_grad():
        writer = open(prediction_file_path, "w")
        for batch in tqdm(dataloader):
            batch = to_cuda(batch, device)
            batch[
                "img"
            ] = horizon_plugin_pytorch.nn.bgr_to_yuv444.bgr_to_yuv444(
                batch["img"].permute(0, 3, 1, 2), False
            )
            batch["img"] = (batch["img"].float() - 128.0) / 128.0
            outputs = model(batch)
            outputs = outputs[1]
            hm = outputs.face_detection_pred_hm.sigmoid()
            wh = outputs.face_detection_pred_wh
            maxp = torch.nn.functional.max_pool2d(
                hm, (3, 3), stride=1, padding=(1, 1)
            )
            keep = (maxp == hm).float()
            hm = hm * keep
            # 1 x 1 x 48x 240
            hm = hm.reshape(hm.size(0), -1)  # 1 x (48x240)
            wh = wh.permute(0, 2, 3, 1).reshape(hm.size(0), -1, 2)

            scores, indices = hm.topk(k=40, dim=1)  # 0 - 48 x 240 - 1
            wh = torch.stack(
                [wh_[index] for wh_, index in zip(wh, indices)], dim=0
            )
            center = torch.stack(
                [
                    indices % 240,
                    torch.div(indices, 240, rounding_mode="trunc"),
                ],
                dim=-1,
            )
            bboxes = torch.cat((center - wh, center + wh), dim=-1)
            bboxes = bboxes * 4
            scales = batch["scale"]
            offsets = batch["offset"]
            img_names = batch["img_name"]
            for img_name, scale, offset, score, bbox in zip(
                img_names, scales, offsets, scores, bboxes
            ):
                bbox = (bbox + offset) * scale
                keep = nms(bbox, score, 0.5)
                bbox = bbox[keep].cpu().numpy()
                score = score[keep].cpu().numpy()
                predictions = []
                for bbox_, score_ in zip(bbox, score):
                    predictions.append(
                        {
                            "bbox": bbox_.tolist(),
                            "bbox_score": score_.tolist(),
                            "attrs": {},
                        }
                    )
                img_name = img_name.split("/")
                face_json = {}
                face_json["image_key"] = os.path.join(
                    img_name[-2], img_name[-1]
                )
                face_json["face"] = predictions
                face_json = str(face_json).replace("'", '"')
                writer.write(face_json)
                writer.write("\n")
    return prediction_file_path


def plate_evaluation(model, dataset_id, device):
    dirname = f"/horizon-bucket/auto_eval/adas_eval/eval_platform/fs/{dataset_id}/datasets/*"  # noqa
    json_file = glob(os.path.join(dirname, "data.json"))[0]
    local_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prediction_file_path = f"plate_{local_time}_.json"
    dataset = AnonymizationEvalDataset(json_file)
    dataloader = data.DataLoader(
        dataset, batch_size=32, num_workers=8, shuffle=True
    )

    with torch.no_grad():
        writer = open(prediction_file_path, "w")
        for batch in tqdm(dataloader):
            batch = to_cuda(batch, device)
            batch[
                "img"
            ] = horizon_plugin_pytorch.nn.bgr_to_yuv444.bgr_to_yuv444(
                batch["img"].permute(0, 3, 1, 2), False
            )
            batch["img"] = (batch["img"].float() - 128.0) / 128.0
            outputs = model(batch)
            outputs = outputs[2]
            hm = outputs.plate_detection_pred_hm.sigmoid()
            wh = outputs.plate_detection_pred_wh
            maxp = torch.nn.functional.max_pool2d(
                hm, (3, 3), stride=1, padding=(1, 1)
            )
            keep = (maxp == hm).float()
            hm = hm * keep
            hm = hm.reshape(hm.size(0), -1)
            wh = wh.permute(0, 2, 3, 1).reshape(hm.size(0), -1, 2)

            scores, indices = hm.topk(k=40, dim=1)
            wh = torch.stack(
                [wh_[index] for wh_, index in zip(wh, indices)], dim=0
            )
            center = torch.stack(
                [
                    indices % 240,
                    torch.div(indices, 240, rounding_mode="trunc"),
                ],
                dim=-1,
            )
            bboxes = torch.cat((center - wh, center + wh), dim=-1)
            bboxes = bboxes * 4
            scales = batch["scale"]
            offsets = batch["offset"]
            img_names = batch["img_name"]
            for img_name, scale, offset, score, bbox in zip(
                img_names, scales, offsets, scores, bboxes
            ):
                bbox = (bbox + offset) * scale
                keep = nms(bbox, score, 0.5)
                bbox = bbox[keep].cpu().numpy()
                score = score[keep].cpu().numpy()
                predictions = []
                for bbox_, score_ in zip(bbox, score):
                    predictions.append(
                        {
                            "bbox": bbox_.tolist(),
                            "bbox_score": score_.tolist(),
                            "attrs": {},
                        }
                    )
                img_name = img_name.split("/")
                plate_json = {}
                plate_json["image_key"] = os.path.join(
                    img_name[-2], img_name[-1]
                )
                plate_json["plate"] = predictions
                plate_json = str(plate_json).replace("'", '"')
                writer.write(plate_json)
                writer.write("\n")
    return prediction_file_path


if __name__ == "__main__":
    cfg_file = "projects/mono/real3d_multitask/multitask.py"

    device = torch.device("cuda:1")
    pretrained_ckpt = "http://fm-xiang-yan.alitrain.hogpu.cc/plat_gpu/hat_mono_real3d_vehicle_detection_face_detection_plate_detection-20221206-195640/output/models/mono_real3d_multitask/qat-checkpoint-last.pth.tar"  # noqa
    cfg = Config.fromfile(cfg_file)
    model = cfg["deploy_model"]
    model = build_from_registry(model)
    horizon_plugin_pytorch.march.set_march(
        horizon_plugin_pytorch.march.March.BERNOULLI2
    )
    if pretrained_ckpt.find("float") != -1:
        checkpoint = load_checkpoint(pretrained_ckpt, check_hash=False)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.fuse_model()
        model.set_qconfig()
        model = horizon_plugin_pytorch.quantization.prepare_qat(
            model, inplace=False
        )
        checkpoint = load_checkpoint(pretrained_ckpt)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model = horizon_plugin_pytorch.quantization.convert(
            model, inplace=False
        )

    model = model.eval().to(device)
    prediction_name = "mono_real3d_multitask_mr"
    tags = [
        "ID:221148",
        "multitask",
        "int_infer",
        "bs16",
        "max_depth_80",
    ]
    token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjIyNTg2MTM0OTMsIlRva2VuVHlwZSI6ImxkYXAiLCJVc2VyTmFtZSI6InhpYW5nLnlhbiIsIk9yZ2FuaXphdGlvbiI6InJlZ3VsYXItZW5naW5lZXIiLCJPcmdhbml6YXRpb25JRCI6MX0.xpXmFEEQ-jpCMBGwOVA6HTjhKpA6YK-Qcqk2FYROVRjrM8Kgn4sksWerUinohmzcaDsb0PluLfRbJ-DZZMF34EWofbk2GJpDUowGnp9XwhtCyQA-tOZPPbfwxPpxgKDNKdMKQWFpxhZ5d3HUAUD2dNSonkwgmYUZP5TOYQkt4lgwBX8myoo1OA_SWcm7IXPFwWbL5buqGidJr36uy5QgjIcBACobXpDehUEGNNTJYxOXR1LfrTuPA-OKxnCFUh8UeWCduufBSLUgeB8JNIALudN5sQY_sMB5OnlGK19kSJukjIgp9H_ZFxwN6rudHFh-ZREJPm-Pg_wUdfpQekQM1w"  # noqa

    # # -------------------------------- Face detection ---------------------------------- """  # noqa
    # dataset_ids = [
    #     6037080,
    # ]
    # for dataset_id in dataset_ids:
    #     prediction_file_path = face_evaluation(model, dataset_id, device)
    #     submit_evaluation(
    #         token, dataset_id, prediction_name, prediction_file_path, tags
    #     )

    # # -------------------------------- Plate detection ---------------------------------- """  # noqa
    # dataset_ids = [6031457]
    # for dataset_id in dataset_ids:
    #     prediction_file_path = plate_evaluation(model, dataset_id, device)
    #     submit_evaluation(
    #         token, dataset_id, prediction_name, prediction_file_path, tags
    #     )
    # # -------------------------------- Vehicle detection ---------------------------------- """  # noqa
    dataset_ids = [
        6038653,
        6038635,
        6038634,
        6038650,
        6038652,
        6038633,
        6036738,
        6036724,
    ]
    for dataset_id in dataset_ids:
        prediction_file_path = vehicle_evaluation(model, dataset_id, device)
        submit_evaluation(
            token, dataset_id, prediction_name, prediction_file_path, tags
        )
