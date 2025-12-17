import argparse
import os
import random

import horizon_plugin_pytorch as horizon
import numpy as np
import torch

from hat.registry import RegistryContext, build_from_registry
from hat.utils.config import Config

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def prepare_test(file_path, model_address):
    model_cfg = Config.fromfile(file_path)
    horizon.march.set_march(model_cfg.march)
    model_cfg["int_infer_trainer"]["model_convert_pipeline"]["converters"][1][
        "checkpoint_path"
    ] = model_address
    with RegistryContext():
        trainer = model_cfg.int_infer_trainer
        trainer = build_from_registry(trainer)
    return trainer.model.eval()


def convert_res_to_dict(outputs):
    single_frame_key = [
        "lane_segmentation",
        "default_segmentation",
        "face_detection",
        "vehicle_plate_detection",
    ]
    merged = {}
    res_dict = {}
    for output in outputs:
        for field in output._fields:
            results = getattr(output, field)
            for i, res in enumerate(results):
                # replace batch index by task result index
                key = "%s_%d" % (field[:-1], i)
                if any([k for k in single_frame_key if k in key]):
                    if key not in merged:
                        merged[key] = []
                    merged[key].append(res.float())
                else:
                    res_dict[key] = res.float()
    for key in merged:
        res_dict[key] = torch.stack(merged[key], dim=0).squeeze()
    return res_dict


def int_model_verify_split(model_address, cfg_path, model_setting):
    os.environ["HAT_PILOT_MODEL_SETTING"] = model_setting
    cfg_dir = os.path.join(os.path.dirname(__file__), "../../configs")
    os.environ["HAT_TRAINING_STEP"] = "int_infer"
    os.environ["HAT_PILOT_BEV_SPLIT"] = "0"
    model_cat = prepare_test(
        os.path.join(cfg_dir, cfg_path, "multitask.py"), model_address
    )
    os.environ["HAT_PILOT_BEV_SPLIT"] = "1"
    model_split = prepare_test(
        os.path.join(cfg_dir, cfg_path, "multitask.py"), model_address
    )
    model_cfg = Config.fromfile(os.path.join(cfg_dir, cfg_path, "models.py"))
    input_hw = model_cfg["input_hw"]
    warp_sizes = model_cfg["warp_sizes"]
    view_num = model_cfg["view_num"]
    inputs = {
        "img_%d" % i: torch.randn(1, 3, input_hw[0], input_hw[1])
        for i in range(view_num)
    }
    inputs.update(
        {
            "homo_offset_%d" % i: torch.randn(1, s[0], s[1], 2)
            for i, s in enumerate(warp_sizes)
        }
    )
    output_split = model_split(inputs)
    output_cat = model_cat(inputs)
    res_split = convert_res_to_dict(output_split)
    res_cat = convert_res_to_dict(output_cat)
    for key in res_cat:
        print(key, " shape:", res_split[key].size())
        diff = (res_split[key] - res_cat[key]).abs().mean().item()
        mean1 = res_split[key].abs().mean().item()
        mean2 = res_cat[key].abs().mean().item()
        print(
            "diff: %.6f, ratio: %.6f, mean split: %.6f, mean cat: %.6f"
            % (
                diff,
                diff * 2 / (mean1 + mean2) if mean1 != 0 and mean2 != 0 else 0,
                mean1,
                mean2,
            )
        )
        assert torch.all(res_split[key] == res_cat[key]), "value diff exists"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-address",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--cfg-path",
        type=str,
        default="resize_2_bev",
    )
    parser.add_argument(
        "--model-setting",
        type=str,
        default="hc23_x3c_day",
    )
    args = parser.parse_args()
    int_model_verify_split(
        args.model_address, args.cfg_path, args.model_setting
    )
