# Copyright (c) Horizon Robotics. All rights reserved.
import argparse
import copy
import struct

import numpy as np

from hat.utils.config import Config

INT_SIZE = 4
FLOAT_SIZE = 4

# usage: ./float2int8.py input_emb.bin output_emb.bin scale


def float2int8(deploy_config: dict):
    # load embedding
    emb = None
    with open(deploy_config["float_emb"], "rb") as ifp:
        tmp = ifp.read(INT_SIZE)
        voc_size = struct.unpack("i", tmp)[0]
        tmp = ifp.read(INT_SIZE)
        emb_size = struct.unpack("i", tmp)[0]
        print("voc %d, emb %d" % (voc_size, emb_size))
        try:
            tmp = ifp.read(voc_size * emb_size * FLOAT_SIZE)
            emb = struct.unpack("%df" % (voc_size * emb_size), tmp)
        except Exception:
            print("fail to read %d * %d float values" % (voc_size, emb_size))
            exit(-1)
    emb = [round(v / deploy_config["scale"]) for v in emb]  # 量化后四舍五入
    emb = np.clip(emb, -128, 127)
    emb_fix = list(np.array(emb, dtype="int"))

    print(emb_fix[512 * 5 : 512 * 6])

    # output
    data = [voc_size, emb_size] + emb_fix
    with open(deploy_config["quantized_emb"], "wb") as ofp:
        tmp = struct.pack("2i%d%s" % (voc_size * emb_size, "b"), *data)
        ofp.write(tmp)

    print(
        "Conversion finished! Quantized embedding is saved in: ",
        deploy_config["quantized_emb"],
    )


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
        "--input_emb",
        type=str,
        help="the path of float embedding",
        # default="",
    )
    parser.add_argument(
        "--output_emb",
        type=str,
        help="the path to save quantized embedding",
        # default="",
    )
    parser.add_argument(
        "--scale",
        type=float,
        help="the quantization parameters",
        default=0.0078125,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    deploy_config = (
        copy.deepcopy(cfg.deploy_cfg) if cfg.deploy_cfg is not None else {}
    )
    if args.input_emb is not None:
        deploy_config["float_emb"] = args.input_emb
    if args.output_emb is not None:
        deploy_config["quantized_emb"] = args.output_emb
    if args.scale is not None:
        deploy_config["scale"] = args.scale
    print(deploy_config)

    float2int8(deploy_config)
