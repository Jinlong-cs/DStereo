import argparse
import glob
import inspect
import itertools
import json
import logging
import os
import sys
from typing import Iterable, Optional

import cv2
import horizon_plugin_pytorch as horizon
import numpy as np
import PIL.Image as pil
import torch
import torch.distributed as dist
import torch.nn.functional as F
from psd_validation import psd_visualization
from tqdm import tqdm

from hat.metrics.mean_iou import MeanIOU
from hat.models.task_modules.ipm_psd.psd_postprocess import PSDPostprocess
from hat.registry import build_from_registry
from hat.utils.checkpoint import load_checkpoint, load_state_dict
from hat.utils.config import Config
from projects.superparking.tools.parsing.parsing_vis import (
    draw_all,
    super_parking_ipm_color_list,
)

sys.setrecursionlimit(100000)
# single task head prediction name
task_pred_name = {"parsing": "parsing_head_predict", "psd": "psd_preds"}


def to_numpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, pil.Image):
        return np.array(x)[:, :, ::-1].copy()
    elif isinstance(x, list):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise ValueError(f"Unknown data type: {type(x)}")


class Inference:

    """Simple fsd inference engine.

    Args:

        config: Input training config file path.
        step: Training step, only support {TRAINING_STEPS}.
        data_path_list: Input image path, only support dir
        pt: Input torchscript file path,int infer stage model
        ckpt: Input checkpoint file path, support http,float&qat stage model
        task: Inference task option, must in ['parsing', 'psd', 'all'].
        status: Evaluation stage control,must in ["val", "test", "aidi_eval"]
        gpu_ids: Index of gpus to use, separated by `,`'.
        save_path: pred visualization directory.
        global_threshold: PSD global head threshold.
        local_threshold: PSD local head threshold.

    """

    default_callbacks = [
        dict(type="PassThroughData"),
    ]

    def __init__(
        self,
        config: str,
        stage: str,
        task: str,
        graph_model_name: str = "model",
        data_paths: Optional[list] = None,
        ckpt: Optional[str] = None,
        status: Optional[str] = None,
        gpu_ids: Optional[str] = None,
        save_path: Optional[str] = None,
        global_threshold: float = 0.18,
        local_threshold: float = 0.25,
    ):
        assert stage in ["float", "qat", "int_infer"]
        assert task in ["parsing", "psd"]
        assert status in ["val", "test", "aidi_eval"]

        # parse gpu_ids
        if gpu_ids and torch.cuda.is_available():
            self.gpu_ids = [int(gpu_id) for gpu_id in gpu_ids.split(",")]
        else:
            self.gpu_ids = []

        # create config and logger
        self.config = Config.fromfile(config)
        self.graph_model_name = graph_model_name
        self.data_paths = data_paths

        # set member
        self.stage = stage
        self.ckpt = ckpt or self.search_checkpoint_path(self.stage)
        if not self.ckpt:
            raise ValueError("Please provide checkpoint path to run inference")

        self.save_path = save_path
        if self.save_path and self.stage != "test":
            logging.warning("Save path is only used in test stage")

        self.global_threshold = global_threshold
        self.local_threshold = local_threshold

        # psd post_process
        self.postprocess = PSDPostprocess(
            input_size=[896, 896],
            downsample_factor_globalslot_feature=32,
            downsample_factor_localjunction_feature=4,
            threshold_globalslot_feature=self.global_threshold,
            threshold_localjunction_feature=self.local_threshold,
            topk_globalslot_feature=50,
            topk_localjunction_feature=30,
            global_kernel_size=3,
            local_kernel_size=5,
            threshold_fuse_distance=55,
            draw_flag=False,
            nms_distance_threshold=20,
        )

        self.task = task
        self.status = status

    def ddp_available(self):
        return dist.is_available() and dist.is_initialized()

    def search_checkpoint_path(self, stage: str):
        if stage == "float":
            return self.config.get("float_checkpoint_path", None)
        elif stage == "qat":
            return self.config.get("qat_checkpoint_path", None)
        elif stage == "int_infer":
            return self.config.get("int_infer_checkpoint_path", None)
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def load_model_params(self):
        config = self.config.get(self.graph_model_name)
        if not config:
            raise ValueError(f"Cannot find {self.graph_model_name} in config")

        model = build_from_registry(config)
        model.eval()
        if self.stage == "qat":
            model = self.convert_qat_model(model)

        model = load_state_dict(
            model,
            load_checkpoint(self.ckpt, map_location="cpu"),
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
        )
        return model

    def convert_qat_model(self, model):
        """Inplace conversion of model.

        Users can also use return value to get converted model.

        Returns:
            Converted model.
        """
        horizon.march.set_march(horizon.quantization.March.BERNOULLI2)

        if isinstance(model, torch.nn.DataParallel):
            model.module.fuse_model()
            model.module.set_qconfig()
        else:
            model.fuse_model()
            model.set_qconfig()

        return model

    def deploy_model(self, model):
        if not self.gpu_ids:
            return model

        model = model.cuda(self.gpu_ids[0])

        if not self.ddp_available():
            return model

        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True, device_ids=self.gpu_ids
        )
        return model

    def create_model(self):
        if self.ckpt.endswith(".pt"):
            model = torch.jit.load(self.ckpt)
            model.eval()
            if self.gpu_ids:
                model = model.cuda(self.gpu_ids[0])  # why only use first gpu?
        else:
            model = self.load_model_params()
            model = self.deploy_model(model)

        return model

    def file2batch(self, file, to_rgb=True):
        image = cv2.imread(file)
        img_name = os.path.basename(file)
        color_space = "bgr"
        if to_rgb:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            color_space = "rgb"
        data = dict()
        data["img_name"] = img_name
        data["img"] = image
        data["ori_img"] = image.copy()
        data["color_space"] = color_space
        data["layout"] = "hwc"
        data["img_shape"] = image.shape
        data["img_id"] = torch.Tensor([[0]])
        data["img_height"] = torch.Tensor([[image.shape[0]]])
        data["img_width"] = torch.Tensor([[image.shape[1]]])

        # transformer
        transforms_cfg = [
            dict(type="ToTensor", to_yuv=True),
            dict(type="ImageNormalize", mean=128.0, std=128.0),
        ]
        transforms = build_from_registry(transforms_cfg)
        for transform in transforms:
            data = transform(data)
        data["img"] = torch.unsqueeze(data["img"], 0).cuda(self.gpu_ids[0])
        data["img_name"] = [data["img_name"]]
        data["ori_img"] = [data["ori_img"]]
        data["layout"] = [data["layout"]]
        data["img_shape"] = torch.unsqueeze(
            torch.tensor(data["img_shape"]), 1
        ).cuda(self.gpu_ids[0])

        return data

    def file2batch_psd(self, data):
        image = data["img"]

        data["img_id"] = torch.Tensor(
            [
                [0],
            ]
        )
        data["img_height"] = torch.Tensor(
            [
                [image.shape[-2]],
            ]
        )
        data["img_width"] = torch.Tensor(
            [
                [image.shape[-1]],
            ]
        )

        data["img"] = data["img"].cuda(self.gpu_ids[0])
        return data

    def model_process(self, model, data, save_path):
        if self.stage == "int_infer":
            data_input = {}
            data_input["img"] = data["img"]
        else:
            data_input = data
        with torch.no_grad():
            preds = model(data_input)
        if self.task == "parsing":
            preds = preds[0][0]  # first task, first output
            parsing_save_path = os.path.join(save_path, "parsing")
            self.parsing_task(preds, data, parsing_save_path)
        elif self.task == "psd":
            preds = preds[1][0]
            psd_save_path = os.path.join(save_path, "psd")
            self.psd_task(preds, data, psd_save_path)
        else:
            parsing_pred = preds[0][0]
            psd_pred = preds[1][0]
            parsing_save_path = os.path.join(save_path, "parsing")
            psd_save_path = os.path.join(save_path, "psd")
            self.parsing_task(parsing_pred, data, parsing_save_path)
            self.psd_task(psd_pred, data, psd_save_path)

    def parsing_task(self, preds, data, save_path):
        preds = preds[0]
        ori_img = data["ori_img"][0]
        ori_h, ori_w = ori_img.shape[0], ori_img.shape[1]
        mask = F.interpolate(
            preds.type(torch.float32), size=(ori_h, ori_w), mode="nearest"
        )
        if self.stage != "int_infer":
            mask = torch.argmax(mask, dim=1)
        else:
            mask = mask.squeeze(dim=1)

        draw_all(
            ori_img,
            mask.squeeze(dim=0),
            data["img_name"][0][:-4],
            save_path,
            color_list=super_parking_ipm_color_list,
        )

    def psd_task(self, preds, data, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        vis_imgs = psd_visualization(preds, data, self.postprocess)
        for idx, name in enumerate(data["img_name"]):
            name = os.path.basename(name)
            cv2.imwrite(os.path.join(save_path, name), vis_imgs[idx])

    def run(self):
        """
        Main loop of inference.
        """
        model = self.create_model()

        # ???????????
        # status is test
        assert self.status == "test"
        assert self.data_paths is not None, f"data_path_list:{self.data_paths}"
        for data in self.data_paths:
            if os.path.isfile(data):
                if data.split(".")[-1] not in ["jpg", "png"]:
                    continue
                batch = self.file2batch(data)
                self.model_process(model, batch, save_path=self.save_path)

            elif os.path.isdir(data):
                files = os.listdir(data)
                for file in tqdm(files):
                    file = os.path.join(data, file)
                    if file.split(".")[-1] not in ["jpg", "png"]:
                        continue
                    batch = self.file2batch(file)
                    self.model_process(model, batch, save_path=self.save_path)
            else:
                assert "data path is not file or dir"


class PSDInference(Inference):
    """Inference for parsing task.

    Args:
        **kwargs: see base class for argument description.
    """

    def __init__(self, **kwargs):
        kwargs["task"] = "psd"
        super().__init__(**kwargs)

    def build_metric(self):
        mtrc_cfg = dict(
            type="PSDMetric",
            global_max_distance=20,
            local_max_distance=5,
            iou_threshold=0.5,
            is_training=False,
        )
        mtrc = build_from_registry(mtrc_cfg)
        return mtrc

    def build_dataset(
        self, data_paths: Iterable[str], transforms: list = None
    ):
        if transforms is None:
            transforms = [
                dict(type="ToTensor", to_yuv=True),
                dict(type="ImageNormalize", mean=128.0, std=128.0),
            ]

        ds = dict(
            type="ConcatDataset",
            datasets=[
                dict(
                    type="PSDSlotDataset",
                    path=data_path,
                    input_size=(896, 896),
                    transforms=transforms,
                )
                for data_path in data_paths
            ],
        )
        return build_from_registry(ds)

    def build_dataloader(self, dataset: torch.utils.data.Dataset):
        from hat.data.collates.collates import collate_psd

        dl = dict(
            type=torch.utils.data.DataLoader,
            dataset=dataset,
            sampler=None,
            batch_size=8,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_psd,
        )
        dl = build_from_registry(dl)
        return dl

    def validate(self, data_paths: Iterable[str]):
        mtrc = self.build_metric()
        dataset = self.build_dataset(data_paths)
        dataloader = self.build_dataloader(dataset)
        model = self.create_model()

        for data in tqdm(dataloader):
            input, labels, img_scene = self._preprocess(data)

            with torch.no_grad():
                output = model(input)

            res = output[1][0]
            res = self.postprocess(res)
            mtrc.update(
                {
                    "labels": labels,
                    "preds_label": res["preds_label"],
                    "img_scene": img_scene,
                }
            )

        names, values = mtrc.get()
        values = [v.cpu().numpy() for v in values]

        print("\nValidation Results:")
        for name, value in zip(names, values):
            print(f"{name}: {value}")

        return names, values

    def infer(
        self, data_paths: Iterable[str], save_path: str = "psd_preds.json"
    ):
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        dataset = self.build_dataset(data_paths)
        dataloader = self.build_dataloader(dataset)
        model = self.create_model()

        with open(save_path, "a") as f:
            for data in tqdm(dataloader):
                input, labels, img_scene = self._preprocess(data)

                with torch.no_grad():
                    output = model(input)

                res = output[1][0] if len(output[0][0]) != 9 else output[0][0]
                res = [x.cpu() for x in res]
                for batch_idx, features in enumerate(zip(*res)):
                    meta_data = {
                        "image_key": data["img_name"][batch_idx],
                        "pred_slots": [],
                    }
                    meta_data = self.postprocess.process(
                        features, meta_data, batch_idx
                    )
                    json.dump(meta_data, f)
                    f.write("\n")

        print(f"save prediction result to {save_path}")

    def _preprocess(self, batch) -> tuple:
        labels = batch.pop("label")
        img_scene = batch["img_scene"]
        batch = self.file2batch_psd(batch)
        batch = batch if self.stage != "int_infer" else {"img": batch["img"]}
        return batch, labels, img_scene


class ParsingInference(Inference):
    def __init__(self, **kwargs):
        kwargs["task"] = "parsing"
        super().__init__(**kwargs)

    def build_metric(self):
        return MeanIOU(seg_class=[str(i) for i in range(21)], ignore_index=255)

    def build_dataset(
        self, data_paths: Iterable[str], transforms: list = None
    ):
        if transforms is None:
            transforms = [
                dict(type="ToTensor", to_yuv=True),
                dict(type="ImageNormalize", mean=128.0, std=128.0),
            ]

        ds = dict(
            type="ConcatDataset",
            datasets=[
                dict(
                    type="DenseboxDataset",
                    data_path=os.path.join(data_path, "val.rec"),
                    anno_path=os.path.join(data_path, "val.json"),
                    task_type="segmentation",
                    rec_idx_file_path=os.path.join(data_path, "val.rec.idx"),
                    transforms=transforms,
                    to_rgb=True,
                )
                for data_path in data_paths
            ],
        )
        return build_from_registry(ds)

    def build_dataloader(self, dataset: torch.utils.data.Dataset):
        dl = dict(
            type=torch.utils.data.DataLoader,
            dataset=dataset,
            sampler=None,
            batch_size=8,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        return build_from_registry(dl)

    def validate(self, data_paths: Iterable[str]):
        mtrc = self.build_metric()
        dataset = self.build_dataset(data_paths)
        dataloader = self.build_dataloader(dataset)
        model = self.create_model()

        for data in tqdm(dataloader):
            data["img"] = data["img"].cuda(self.gpu_ids[0])
            input = {"img": data["img"]} if self.stage == "int_infer" else data
            with torch.no_grad():
                output = model(input)
            output = output[0][0]
            preds = output[0]
            mask = F.interpolate(
                preds.type(torch.float32), size=(896, 896), mode="nearest"
            )
            if self.stage != "int_infer":
                mask = mask.argmax(dim=1)
            mask = mask.squeeze(dim=1)
            target = data["gt_seg"].squeeze(dim=1)
            mtrc.update(target, mask.cpu())
        return mtrc.get()

    def infer(
        self, data_paths: Iterable[str], save_path: str = "parsing_preds"
    ):
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        model = self.create_model()

        for data_path in data_paths:
            if os.path.isfile(data_path):
                batch = self.file2batch(data_path)
                batch = [batch]
            else:
                print(f"Load data from {data_path}")
                jpgs = glob.iglob(data_path + "/**/*.jpg", recursive=True)
                pngs = glob.iglob(data_path + "/**/*.png", recursive=True)
                files = itertools.chain(jpgs, pngs)
                files = filter(
                    lambda x: "_label" not in x and "_render" not in x, files
                )
                files = map(lambda x: os.path.join(data_path, x), files)
                batches = map(self.file2batch, files)

            for batch in tqdm(batches):
                self.model_process(model, batch, save_path)

        print(f"save prediction result to {save_path}")


def get_default_psd_data_paths():
    try:
        from projects.superparking.app.ipm.base import psd_val_paths
    except ImportError:
        psd_val_paths = []

    return psd_val_paths


def get_default_parsing_data_paths(mode="aidi_eval"):
    try:
        from projects.superparking.app.ipm.base import parsing_val_paths
    except ImportError:
        parsing_val_paths = []

    if mode == "aidi_eval":
        return [
            "/horizon-bucket/auto_eval/adas_eval/eval_platform/fs/6039118/datasets/datasets/images"  # noqa
        ]
    else:
        return parsing_val_paths


def calculate_data_scene(data_paths: Iterable[str], infer: Inference):
    from collections import Counter

    ds = infer.build_dataset(data_paths, transforms=[])
    scenes = [x["img_scene"] for x in tqdm(ds)]

    print(f"total length of data: {len(scenes)}")
    print(f"distribution of scenes: {Counter(scenes)}\n")


def main(args):
    tasks = args.task.replace(" ", "").split(",")
    data_paths = getattr(args, "data_path_list", None)

    kwargs = {
        k: getattr(args, k)
        for k in dir(args)
        if k in inspect.signature(Inference.__init__).parameters.keys()
    }
    kwargs.pop("task")

    psd_paths = filter(lambda x: "psd" in x.lower(), data_paths or [])
    parsing_paths = filter(lambda x: "parsing" in x.lower(), data_paths or [])

    parsing_infer, psd_infer = None, None

    if "parsing" in tasks:
        parsing_paths = list(parsing_paths) or get_default_parsing_data_paths()
        parsing_infer = ParsingInference(**kwargs)
    if "psd" in tasks:
        psd_paths = list(psd_paths) or get_default_psd_data_paths(
            mode=args.status
        )
        psd_infer = PSDInference(**kwargs)

    if args.status == "val":
        psd_infer and psd_infer.validate(psd_paths)
        parsing_infer and parsing_infer.validate(parsing_paths)
    elif args.status == "aidi_eval":
        psd_infer and psd_infer.infer(psd_paths)
        parsing_infer and parsing_infer.infer(parsing_paths)


def parse_infer_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="",
        help="inference mode",
    )
    parser.add_argument(
        "--graph-model-name",
        type=str,
        default="model",
        help="graph model name inside the configuration file",
    )
    parser.add_argument(
        "--data_path_list",
        "-d",
        type=str,
        nargs="+",
        default=None,
        help="inference data path",
    )
    parser.add_argument(
        "--stage", type=str, default="int_infer", help="inference mode"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="parsing, psd",
    )
    parser.add_argument(
        "--status",
        type=str,
        default="val",
        help="val or test or aidi_eval",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--global-threshold",
        type=float,
        default=0.18,
    )
    parser.add_argument(
        "--local-threshold",
        type=float,
        default=0.25,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_infer_args()
    task_log = "./IPM_" + args.task + ".log"
    logging.basicConfig(filename=task_log, level=logging.DEBUG)
    main(args)
