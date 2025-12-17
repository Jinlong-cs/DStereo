"""浮点模型推理流程代码."""

__author__ = "ruosong.li@horizon.ai"
__date__ = "2023/01/16"
__copyright__ = "Copyright (C) 2023 Horizon Robotics, Inc."

import argparse
import json
import math
import os

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from hat.core.nlu.nlu_utils import transfer_batch_sample
from hat.data.datasets.nlu_dataset import NluBasicTokenizer, NluLabelProcessor
from hat.models.backbones.nlu_tcn import NluMultiTaskBackbone
from hat.models.structures.nlu import NluModel
from hat.utils.config import Config

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
RUNNING_PATH = os.getcwd()


def parse_args():
    """Obtain configuration from .cfg file.

    Returns:
        dict: Python dict of configuration for inferencing trained model.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="ptq or qat config file path",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="the path of input file,each line contain one query.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="name of output file.",
        default="float_infer_out.txt",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        help="batch size.",
        default=10,
    )
    cmd_args = parser.parse_args()
    cfg = Config.fromfile(cmd_args.config)

    args = {}
    args["dict_path"] = cfg.vocab_path
    args["label_path"] = cfg.label_path
    args["tcn_levels"] = cfg.tcn_levels
    args["query_file"] = os.path.join(RUNNING_PATH, cmd_args.input)
    args["ckpt_file"] = os.path.join(
        cfg.ckpt_dir, "float-checkpoint-best.pth.tar"
    )
    args["output_file"] = os.path.join(cfg.ckpt_dir, cmd_args.output)
    args["batch_size"] = cmd_args.batch_size

    return args


def load_query_2_batch(query_path, batch_size=10):
    """Load and transform the query into batches.

    Args:
        query_path (str): Path to query data file.
        batch_size (int): Query num in a batch.

    Returns:
        List[List]: A list contains batch data (lists of string samples),
                    eg: [['打开车窗', '打开车窗窗', '开一下空调', ...], [...]].

    """
    with open(query_path, "r", encoding="utf-8") as f:
        all_querys = f.readlines()
    all_querys = [
        item.replace("\n", "") for item in all_querys if item.replace("\n", "")
    ]
    all_batches = []
    batch_num = math.ceil(len(all_querys) / batch_size)
    for i in range(batch_num):
        all_batches.append(all_querys[i * batch_size : (i + 1) * batch_size])
    return all_batches


class MultiTaskPredictor:
    """Multi-task predictor.

    MultiTaskPredictor can predict domain, intent and slot at
    one inference from a trained MultiTaskModel model.

    """

    def __init__(
        self, model_ckpt_path, tokenizer, label_processor, tcn_levels, device
    ):
        """Init MultiTaskPredictor class.

        Args:
            model_ckpt_path (str): Path to trained model.
            tokenizer (utils.tokenizer.BasicTokenizer):
                Provide method to transform word list to token-index list.
            label_processor (utils.label_processor.LabelProcessor):
                Process the index to original label.
            tcn_levels (int): Number of Backbone in TCN model.
            device (torch.device): Run on cuda or cpu.

        """
        self.args = args
        backbone = NluMultiTaskBackbone(
            tokenizer,
            label_processor,
            tcn_levels=tcn_levels,
        )
        model = NluModel(backbone, nn.CrossEntropyLoss(reduction="none"))
        ckpt_full = torch.load(model_ckpt_path)
        model.load_state_dict(ckpt_full["state_dict"])

        self.model = model
        self.model.to(device)
        self.device = device
        self.tokenizer = tokenizer
        self.label_processor = label_processor

    def predict(self, batch_query):
        """Predict results.

        Args:
            batch_query (List[str]): Batch query data.

        Returns:
            List[dict]: List of result dicts, containing domain,
                intent and slots predictions of each query.

        """
        # data preparing
        batch_len = len(batch_query)

        batch_ids = [
            self.tokenizer.tokenize(["<dom>", "<int>"] + list(query))
            for query in batch_query
        ]
        # 去掉<int> <dom>
        throw_tokens = 2

        valid_lengths = [len(item) for item in batch_ids]
        batch_pad_ids = pad_sequence(
            [torch.tensor(ids) for ids in batch_ids], batch_first=True
        ).to(self.device)
        batch_pad_masks = pad_sequence(
            [
                torch.tensor([True for i in range(length - throw_tokens)])
                for length in valid_lengths
            ],
            batch_first=True,
        ).to(self.device)
        batch_sample_weights = torch.tensor([1] * batch_len).to(self.device)
        fake_domain_labels = torch.randint(1, (batch_len,)).to(self.device)
        fake_intent_labels = torch.randint(1, (batch_len,)).to(self.device)
        fake_slots_labels = torch.randint(
            1, (batch_len, max(valid_lengths) - throw_tokens)
        ).to(self.device)
        batch_data = {
            "batch_ids": batch_pad_ids,
            "batch_pad_masks": batch_pad_masks,
            "batch_domain_labs": fake_domain_labels,
            "batch_intent_labs": fake_intent_labels,
            "batch_ner_labs": fake_slots_labels,
            "batch_sample_weights": batch_sample_weights,
        }
        # predict
        self.model.eval()
        preds, _, _ = self.model(batch_data)
        # output reformat
        cur_domain_pred = preds[0]
        cur_intent_pred = preds[1]
        cur_slots_pred = preds[2]
        pred_reformat = transfer_batch_sample(
            batch_query, cur_domain_pred, cur_intent_pred, cur_slots_pred
        )
        return pred_reformat


def inference(args):
    """Inference model.

    Args:
        args (dict): Configuration for inferencing trained model.

    """
    tokenizer = NluBasicTokenizer(args["dict_path"])
    label_processor = NluLabelProcessor(args["label_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = MultiTaskPredictor(
        args["ckpt_file"],
        tokenizer,
        label_processor,
        args["tcn_levels"],
        device,
    )

    all_batches = load_query_2_batch(args["query_file"], args["batch_size"])

    pred_all = []
    for batch_id in tqdm(range(len(all_batches))):
        batch_query = all_batches[batch_id]
        pred_all.extend(predictor.predict(batch_query))
    print("finish inferencing!")

    with open(
        args["output_file"],
        "w",
        encoding="utf-8",
    ) as f:
        for pred in pred_all:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print("output file path: {}".format(args["output_file"]))


if __name__ == "__main__":
    args = parse_args()
    inference(args)
