"""psd validation script.

Author: @chunyu.bi

TODO: This script is used to run ipm int evalution successfully. It shoule be
deprecated after the ipm adopts the new evaluation method.
"""
import argparse
import inspect
import os
import sys
from typing import Optional

import cv2
import horizon_plugin_pytorch as horizon
import numpy as np
import PIL.Image as pil
import torch
import torch.distributed as dist
import torch.nn as nn
from termcolor import cprint
from tqdm import tqdm

from hat.registry import build_from_registry
from hat.utils.checkpoint import load_checkpoint, load_state_dict
from hat.utils.config import Config

sys.setrecursionlimit(100000)


def psd_visualization(preds, data, postprocess):

    (
        global_features,
        global_offsets,
        global_occupancys,
        global_slot_types,
        global_directions,
        local_features,
        local_offsets,
        local_sline_angles,
        local_point_types,
    ) = preds

    global_features = global_features.cpu()
    global_offsets = global_offsets.cpu()
    global_occupancys = global_occupancys.cpu()
    global_slot_types = global_slot_types.cpu()
    global_directions = global_directions.cpu()
    local_features = local_features.cpu()
    local_offsets = local_offsets.cpu()
    local_sline_angles = local_sline_angles.cpu()
    local_point_types = local_sline_angles.cpu()
    meta_data = {}
    meta_data["ori_img"] = data["ori_img"]
    for batch_idx, features in enumerate(
        zip(
            global_features,
            global_offsets,
            global_occupancys,
            global_slot_types,
            global_directions,
            local_features,
            local_offsets,
            local_sline_angles,
            local_point_types,
        )
    ):
        img_name = data["img_name"][batch_idx]
        # print("{} was processed.".format(img_name))
        meta_data["pred_slots"] = []
        # meta_data["save_path"] = save_path
        meta_data = postprocess.process(features, meta_data, batch_idx)
        img_name = os.path.basename(img_name)
    return meta_data["visualized_results"]


class Inference:
    r"""
    Simple fsd inference engine.

    Args:

        config: Input training config file path.
        step: Training step, only support {TRAINING_STEPS}.
        data: Input image path, only support dir
        pt: Input torchscript file path.
        ckpt: Input checkpoint file path, support http.
        task: Inference task option, must in ['parsing', 'psd', 'all'].
        gpu_ids: Index of gpus to use, separated by `,`'.
        save_path: Visualization directory.
        global_threshold: PSD global head threshold.
        local_threshold: PSD local head threshold.
        vis: Whether to save visualization results.


    """

    default_callbacks = [
        dict(type="PassThroughData"),
    ]

    def __init__(
        self,
        config: str,
        step: str,
        data: Optional[str] = None,
        pt: Optional[str] = None,
        ckpt: Optional[str] = None,
        gpu_ids: Optional[str] = None,
        save_path: Optional[str] = None,
        global_threshold=0.18,
        local_threshold=0.25,
        vis=False,
    ):
        assert step in ["float", "qat", "int_infer"]
        # parse gpu_ids
        if gpu_ids and torch.cuda.is_available():
            self.gpu_ids = [int(gpu_id) for gpu_id in gpu_ids.split(",")]
        else:
            self.gpu_ids = []

        # create config and logger
        if step != "int_infer":
            self.cfg = Config.fromfile(config)
        self.data = data
        # set member
        self.step = step
        self.pt = pt
        self.ckpt = ckpt
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.vis = vis

        self.global_threshold = global_threshold
        self.local_threshold = local_threshold

        if self.vis:
            postprocess_cfg = dict(
                type="PSDPostprocess",
                input_size=[896, 896],
                downsample_factor_globalslot_feature=32,
                downsample_factor_localjunction_feature=4,
                threshold_globalslot_feature=global_threshold,
                threshold_localjunction_feature=local_threshold,
                topk_globalslot_feature=50,
                topk_localjunction_feature=30,
                threshold_occupancy_feature=0.5,
                threshold_junction_type_feature=0.5,
                threshold_fuse_distance=30,
                global_kernel_size=3,
                local_kernel_size=5,
                draw_flag=True,
            )
            self.postprocess = build_from_registry(postprocess_cfg)

        self.get_ckpt_from_config_if_need()

    @property
    def ddp(self):
        return dist.is_available() and dist.is_initialized()

    def get_ckpt_from_config_if_need(self) -> None:
        """
        Get checkpoint from config when args.ckpt is "".
        """
        if self.ckpt or self.pt:
            return
        cprint("args.ckpt is None, find from config", "yellow")
        float_checkpoint_path = self.cfg.get("float_checkpoint_path", None)
        qat_checkpoint_path = self.cfg.get("qat_checkpoint_path", None)
        if self.step == "float":
            if float_checkpoint_path:
                self.ckpt = float_checkpoint_path
            else:
                cprint(
                    "float_checkpoint_path not found in config, "
                    "please provide float checkpoint",
                    "red",
                )
                sys.exit()
        elif self.step == "qat":
            if qat_checkpoint_path:
                self.ckpt = qat_checkpoint_path
            else:
                cprint(
                    "qat_checkpoint_path not found in config, "
                    "please provide qat checkpoint",
                    "red",
                )
                sys.exit()
        else:
            raise NotImplementedError

    def to_numpy(self, x):
        """
        Convert data to numpy.ndarray.
        """
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

    def create_model(self):
        """
        Create model from config.
        """
        model_cfg = self.cfg.test_model
        model = build_from_registry(model_cfg)
        model.eval()

        if self.step == "qat":
            horizon.march.set_march(horizon.quantization.March.BERNOULLI2)
            if isinstance(model, nn.DataParallel):
                model.module.fuse_model()
                model.module.set_qconfig()
            else:
                model.fuse_model()
                model.set_qconfig()
            horizon.quantization.prepare_qat(model, inplace=True)

        ckpt_dict = load_checkpoint(
            self.ckpt,
            map_location="cpu",
        )

        load_state_dict(
            model,
            ckpt_dict["state_dict"],
            allow_miss=True,
            ignore_extra=True,
            verbose=1,
        )  # set 2 to debug load state dict problem

        if self.gpu_ids:
            model = model.cuda(self.gpu_ids[0])

        if self.ddp:
            model = nn.parallel.DistributedDataParallel(
                model, find_unused_parameters=True, device_ids=self.gpu_ids
            )

        return model

    def load_model(cfg, model, logger):
        solver = cfg["validation"]
        if solver["pretrained_model"] is not None:
            checkpoint = torch.load(
                solver["pretrained_model"],
                map_location="cpu",
            )
            model = torch.nn.DataParallel(model)
            model.load_state_dict(checkpoint["state_dict"])
            model = model.module
            logger.info(
                "load pretrained model from %s" % (solver["pretrained_model"])
            )
        return model

    def create_pt_model(self):
        assert self.step == "int_infer" and self.pt is not None
        model = torch.jit.load(self.pt)
        model.eval()
        if self.gpu_ids:
            model = model.cuda(self.gpu_ids[0])
        return model

    def file2batch(self, data):
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

    def psd_task(self, model, save_path=None):
        from hat.data.collates.collates import collate_psd

        metric_cfg = dict(
            type="PSDMetric",
            input_size=[896, 896],
            global_threshold=self.global_threshold,
            global_topk=50,
            global_downsample_factor=32,
            global_max_distance=20,
            global_kernel_size=3,
            local_threshold=self.local_threshold,
            local_topk=30,
            local_downsample_factor=4,
            local_max_distance=5,
            local_kernel_size=7,
            threshold_fuse_distance=30,
            iou_threshold=0.5,
            is_training=False,
            draw_flag=False,
        )
        psd_metric = build_from_registry(metric_cfg)
        val_dataset = dict(
            type="PSDSlotDataset",
            path=self.data,
            input_size=(896, 896),
            transforms=[
                dict(type="ToTensor", to_yuv=True),
                dict(type="ImageNormalize", mean=128.0, std=128.0),
            ],
        )
        val_data_loader = dict(
            type=torch.utils.data.DataLoader,
            dataset=val_dataset,
            sampler=None,
            batch_size=8,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_psd,
        )
        dataloader = build_from_registry(val_data_loader)
        for data in tqdm(dataloader):
            labels = data["label"]
            img_scene = data["img_scene"]
            data.pop("label")
            data = self.file2batch(data)
            if self.step == "int_infer":
                data_input = {}
                data_input["img"] = data["img"]
            else:
                data_input = data
            with torch.no_grad():
                outputs = model(data_input)
            outputs = outputs[1][0]
            if self.vis:
                vis_imgs = psd_visualization(outputs, data, self.postprocess)
                for idx, name in enumerate(data["img_name"]):
                    print("{} was processed.".format(name))
                    name = os.path.basename(name)
                    cv2.imwrite(
                        os.path.join(self.save_path, name), vis_imgs[idx]
                    )

            psd_metric.update(
                dict(label=labels, preds=outputs, img_scene=img_scene)
            )
        total_names, total_values = psd_metric.get()
        print(total_names)
        print(total_values)

    def run(self):
        """
        Main loop of inference.
        """

        if self.step == "int_infer":
            model = self.create_pt_model()
        else:
            model = self.create_model()

        self.psd_task(model)


def main(args):
    parameters = inspect.signature(Inference.__init__).parameters.keys()
    kwargs = {k: getattr(args, k) for k in dir(args) if k in parameters}
    engine = Inference(**kwargs)
    engine.run()


def parse_args1():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="",
        help="inference mode",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default="",
        help="inference data path",
    )
    parser.add_argument(
        "--step", type=str, default="float", help="inference mode"
    )
    parser.add_argument(
        "--pt",
        type=str,
        default="",
    )
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="1",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="tmp_multitask_vis/psd-v1.0.0",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="whether do visualization",
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
    args = parse_args1()
    main(args)
