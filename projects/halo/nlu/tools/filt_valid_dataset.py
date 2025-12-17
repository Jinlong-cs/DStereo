"""基于nlu_dataset,统计有效数据，过滤无效的数据样本."""

__author__ = "ruosong.li@horizon.ai"
__date__ = "2022/12/19"
__copyright__ = "Copyright (C) 2022 Horizon Robotics, Inc."

import argparse
import json
import os
import sys

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_PATH, "../../../../"))
from hat.data.datasets.nlu_dataset import (  # noqa: E402
    NluBasicTokenizer,
    NluLabelProcessor,
    NluMultiTaskDataset,
)
from hat.utils.config import Config  # noqa: E402


def parse_args():
    """输入参数解析."""
    parser = argparse.ArgumentParser(
        description="Filt invalid data sample.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="deploy config file path",
    )
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        required=True,
        help="jfs path of input dataset,example: file:///jfs-hdfs/users/xxx",
    )
    parser.add_argument(
        "-lo",
        "--log_file",
        type=str,
        default="data_check.log",
        help="output path of data stat log",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default="valid_dataset.json",
        help="output path of nlu testset",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    input_path = args.input_path
    max_query_length = cfg["max_query_length"]
    save_path = os.path.join(CURRENT_PATH, "./files")
    log_file = os.path.join(save_path, args.log_file)
    output_path = os.path.join(save_path, args.output_file)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    tokenizer = NluBasicTokenizer(cfg["vocab_path"])
    label_processor = NluLabelProcessor(cfg["label_path"])
    nlu_dataset = NluMultiTaskDataset(
        input_path, tokenizer, label_processor, max_query_length, log_file
    )
    valid_data_all = nlu_dataset.data_all
    with open(output_path, "w", encoding="utf-8") as f:
        for data in valid_data_all:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
