# Copyright (c) Horizon Robotics. All rights reserved.
import argparse
import copy
import os
import struct

import torch

from hat.utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="deploy config file path",
    )
    parser.add_argument(
        "--extracted_ckpt",
        type=str,
        help="the path of ckpt",
        # default="./output/nlu_model/nlu_model/qat-checkpoint-best.pth.tar",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="the path to save bin file",
        # default="./output/nlu_model/nlu_model/qat-layers",
    )

    args = parser.parse_args()
    return args


def extract_layers(deploy_config: dict):
    model = torch.load(deploy_config["extracted_ckpt"])["state_dict"]
    # 根据模型算子名称调整"layers"
    layers = [
        "backbone.embedding.weight",
        "backbone.crf.start_transitions",
        "backbone.crf.end_transitions",
        "backbone.crf.transitions",
    ]
    names = [
        "embedding.float.weight",
        "crf.start_transitions",
        "crf.end_transitions",
        "crf.transitions",
    ]
    for i in range(len(layers)):
        if not os.path.exists(deploy_config["layers_path"]):
            os.makedirs(deploy_config["layers_path"])
        save_path = os.path.join(
            deploy_config["layers_path"], names[i] + ".bin"
        )
        cur_param = model[layers[i]].cpu().detach().numpy()
        cur_shape = cur_param.shape
        cur_param = cur_param.tolist()
        with open(save_path, "wb") as f:
            assert len(cur_shape) <= 2
            if len(cur_shape) == 2:
                f.write(struct.pack("2i", cur_shape[0], cur_shape[1]))
                for i in range(cur_shape[0]):
                    for j in range(cur_shape[1]):
                        f.write(struct.pack("f", cur_param[i][j]))
            else:
                f.write(struct.pack("i", cur_shape[0]))
                for i in range(cur_shape[0]):
                    f.write(struct.pack("f", cur_param[i]))
    print("Extract done! Layers are saved in: ", deploy_config["layers_path"])


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    deploy_config = (
        copy.deepcopy(cfg.deploy_cfg) if cfg.deploy_cfg is not None else {}
    )
    if args.extracted_ckpt is not None:
        deploy_config["extracted_ckpt"] = args.extracted_ckpt
    if args.output is not None:
        deploy_config["layers_path"] = args.output

    extract_layers(deploy_config)
