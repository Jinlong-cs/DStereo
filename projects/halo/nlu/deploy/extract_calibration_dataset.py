"""校准集合生成脚本，基于输入样本集进行采样."""
# Copyright (c) Horizon Robotics. All rights reserved.
import json
import math
import os
import sys
from random import shuffle

from hat.utils.config import Config
from hat.utils.filesystem import get_filesystem


def parse_args():
    cfg = Config.fromfile(sys.argv[1])
    args = {
        "input_path": cfg.val_path,
        "output_path": os.path.join(cfg.deploy_dir, "calibration"),
        "calibration_num": cfg.calibration_num,
    }
    return args


if __name__ == "__main__":
    args = parse_args()
    cwd = os.getcwd()
    save_path = os.path.join(cwd, args["output_path"])
    jfs = get_filesystem(args["input_path"])
    with jfs.open(args["input_path"], "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    all_data = [json.loads(i) for i in all_lines]
    shuffle(all_data)
    total_num = args["calibration_num"]
    pos_num = math.ceil(total_num / 2)
    neg_num = total_num - pos_num
    pos_data = []
    neg_data = []
    pos_cnt = 0
    neg_cnt = 0
    idx = 0
    while (pos_cnt < pos_num or neg_cnt < neg_num) and idx < len(all_data):
        cur_domain = all_data[idx]["domain"]
        if cur_domain == "other" and neg_cnt < neg_num:
            neg_data.append(all_data[idx])
            neg_cnt += 1
        elif pos_cnt < pos_num:
            pos_data.append(all_data[idx])
            pos_cnt += 1
        idx += 1
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_all = pos_data + neg_data
    query_all = [i["query"] for i in data_all]
    with open(
        os.path.join(save_path, "calibration_set"), "w", encoding="utf-8"
    ) as f:
        for data in data_all:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    with open(
        os.path.join(save_path, "calibration_queries"), "w", encoding="utf-8"
    ) as f:
        for query in query_all:
            f.write(query + "\n")
