"""生成二进制校验数据."""
import os
import sys

import numpy as np
import torch

from hat.data.datasets.nlu_dataset import NluBasicTokenizer, NluLabelProcessor
from hat.models.backbones.nlu_tcn import NluMultiTaskBackbone
from hat.models.structures.nlu import NluModel
from hat.utils.config import Config


def parse_args():
    cfg = Config.fromfile(sys.argv[1])
    args = {
        "dict_path": cfg.vocab_path,
        "label_path": cfg.label_path,
        "ckpt_path": cfg.ckpt_file,
        "max_query_length": cfg.max_query_length,
        "tcn_levels": cfg.tcn_levels,
        "output_path": os.path.join(cfg.deploy_dir, "calibration/bin"),
        "query_path": os.path.join(
            cfg.deploy_dir, "calibration/calibration_queries"
        ),
    }
    return args


if __name__ == "__main__":
    args = parse_args()
    dict_path = args["dict_path"]
    label_path = args["label_path"]
    ckpt_path = args["ckpt_path"]
    query_len = args["max_query_length"]
    tcn_levels = args["tcn_levels"]
    query_path = args["query_path"]
    output_path = args["output_path"]

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    tokenizer = NluBasicTokenizer(dict_path)
    label_processor = NluLabelProcessor(label_path)
    backbone = NluMultiTaskBackbone(
        tokenizer,
        label_processor,
        tcn_levels=tcn_levels,
    )
    model = NluModel(backbone)
    ckpt_full = torch.load(ckpt_path)
    model.load_state_dict(ckpt_full["state_dict"])

    query_list = []
    with open(query_path, "r") as f:
        for line in f:
            data = line.strip()
            if len(data) > query_len - 2:
                print("**Too long query, truncated sentence!**")
                data = data[: query_len - 2]
            query_list.append(data)

    for i, query in enumerate(query_list):
        num_pad = query_len - 2 - len(query)
        query_ids = torch.tensor(
            tokenizer.tokenize(
                ["<dom>", "<int>"] + list(query) + ["<pad>"] * num_pad
            )
        ).unsqueeze(0)
        input_embed = (
            model.backbone.embedding(query_ids).permute(0, 2, 1).unsqueeze(2)
        )
        input_embed_arr = input_embed.cpu().detach().numpy().astype(np.float32)
        input_embed_arr.tofile(os.path.join(output_path, "%s.bin") % str(i))
    print("finished %s query" % (len(query_list)))
