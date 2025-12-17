"""量化前后onnx精度评估以及输出logits余弦相似度."""
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.multiprocessing
from horizon_tc_ui import HB_ONNXRuntime
from torch.utils.data import DataLoader
from tqdm import tqdm

from hat.core.nlu.nlu_utils import (
    get_batch_ner_label,
    get_diff_samples,
    transfer_batch_sample,
)
from hat.data.collates.collates import collate_nlu_with_pad as collate_func
from hat.data.datasets.nlu_dataset import NluBasicTokenizer as BasicTokenizer
from hat.data.datasets.nlu_dataset import NluLabelProcessor as LabelProcessor
from hat.data.datasets.nlu_dataset import (
    NluMultiTaskDataset as MultiTaskDataset,
)
from hat.utils.config import Config
from projects.halo.nlu.deploy.crf_decoder import CRF

torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args():
    """Obtain configuration from .cfg file.

    Returns:
         dict: Python dict of configuration for training model.

    """
    cfg = Config.fromfile(sys.argv[1])
    args = {}
    args["label_path"] = cfg.label_path
    args["layers_path"] = os.path.join(cfg.deploy_dir, "special_layers")
    args["seq_len"] = cfg.max_query_length
    args["original_onnx_file"] = os.path.join(
        cfg.deploy_dir,
        cfg.platform,
        "hat-tcn_original_float_model.onnx",
    )
    args["quantized_onnx_file"] = os.path.join(
        cfg.deploy_dir,
        cfg.platform,
        "hat-tcn_quantized_model.onnx",
    )
    args["evalset_path"] = cfg.deploy_evalset_path
    args["output_path"] = os.path.join(
        cfg.deploy_dir,
        cfg.platform,
        "quantinized_model_eval_report.log",
    )
    args["batch_size"] = 16
    return args


def export_eval_report(save_path, stat_dic):
    """Export model evaluation report.

    Args:
        save_path (str): Path to save the report.
        stat_dic (dict): Python dict of
            statistical results of evaluate stage.

    """
    with open(save_path, "w", encoding="utf-8") as f:
        for k in stat_dic:
            f.write("**" + k + "**" + "\n")
            f.write(stat_dic[k] + "\n")


class OnnxPredictor:
    """Multi-task predictor for onnx model."""

    def __init__(
        self,
        args,
        label_processor,
        onnx_file,
    ):
        """Init OnnxPredictor class.

        Args:
            args (dict): Configuration for evaluating mode.
            label_processor (utils.label_processor.LabelProcessor):
                Process the index to original label.
            onnx_file (str): Path to onnx model.

        """
        self.args = args
        self.model_session = HB_ONNXRuntime(model_file=onnx_file)
        self.label_processor = label_processor
        self.crf = CRF(
            args["layers_path"],
            batch_first=True,
        )

    def _prepare_input_dict(self, input_names, data):
        feed_dict = {}
        for input_name in input_names:
            feed_dict[input_name] = data
        return feed_dict

    def _inference(self, sess, data):
        feed_dict = self._prepare_input_dict(sess.input_names, data)
        outputs = sess.run_feature(
            sess.output_names, feed_dict, input_offset=0
        )
        return outputs[0], outputs[1], outputs[2]

    def predict(self, input_emb, query, valid_length):
        """Predict results.

        The predictor can only process batch_size=1 situation.

        Args:
            input_emb (np.array): Embeddings of input sequence
                 (1=batch_size, C, 1, W=seq_len).
            query (list): List of input query.
            valid_length (int): Valid length of input query without pads.

        Returns:
            numpy.ndarray: Domain output logits of input query.
            numpy.ndarray: Intent output logits of input query.
            numpy.ndarray: Slots output logits of input query.
            list: List of a python dict of formated data.

        """
        domain_out, intent_out, slots_out = self._inference(
            self.model_session, input_emb
        )
        domain_pred_id = np.argmax(domain_out, axis=1)
        intent_pred_id = np.argmax(intent_out, axis=1)
        ner_tag_all_id = self.crf.decode(slots_out)
        ner_pred_id = [
            ner_tag_all_id[i][:valid_length]
            for i in range(len(ner_tag_all_id))
        ]

        cur_domain_pred = [
            self.label_processor.index_2_domain[i] for i in domain_pred_id
        ]
        cur_intent_pred = [
            self.label_processor.index_2_intent[i] for i in intent_pred_id
        ]
        cur_slots_pred = [
            [self.label_processor.index_2_slots[item] for item in seq]
            for seq in ner_pred_id
        ]
        pred_reformat = transfer_batch_sample(
            query, cur_domain_pred, cur_intent_pred, cur_slots_pred
        )
        return domain_out, intent_out, slots_out, pred_reformat


def load_embedding(embedding_path):
    """Load word embeddings from file.

    Args:
        embedding_path (str): Embedding file path.

    Returns:
        list: List of word embedding parameters of entire vocabulary.

    """
    embedding = []
    with open(embedding_path, "rb") as f:
        embedding_shape = struct.unpack("2i", f.read(8))
        print("embedding_shape: {}".format(embedding_shape))
        for i in range(embedding_shape[0]):
            embedding.append([])
            for _ in range(embedding_shape[1]):
                embedding[i].append(struct.unpack("f", f.read(4))[0])
    print("Successfully load embedding!")
    return embedding


def get_pad_embedding(embedding, query_ids, seq_len):
    """Get word embeddings of input query.

    Args:
        embedding (list): Word embedding parameters of entire vocabulary.
        query (list): List of query word ids.
        seq_len (int): Max query length.

    Returns:
        list: List of word embeddings of input query.

    """
    input_emb = []
    if len(query_ids) < seq_len:
        interval = seq_len - len(query_ids)
        query_ids.extend([0] * interval)
    if len(query_ids) > seq_len:
        query_ids = query_ids[:seq_len]
    for id in query_ids:
        input_emb.append(embedding[int(id)])
    return input_emb


def get_cos_sim(matrix_a, matrix_b):
    """Calculate cosine similarity of two matrix at row level.

    Args:
        matrix_a (numpy.ndarray): Output logits of original (quantized) model.
        matrix_b (numpy.ndarray): Output logits of quantized (original) model.

    Returns:
        numpy.ndarray: Cosine similarity of output logits.

    """
    cos_sim = (
        (matrix_a * matrix_b).sum(axis=1)
        / (matrix_a * matrix_a).sum(axis=1) ** 0.5
        / (matrix_b * matrix_b).sum(axis=1) ** 0.5
    )
    return cos_sim


def get_stat_info(cos_sim):
    """Get statistical results of cosine similarity for whole dataset.

    Args:
        cos_sim (numpy.ndarray): Cosine similarity of domain/input/slots
            logits of all samples.

    Returns:
        numpy.ndarray: Mean value of cosine similarity.
        numpy.ndarray: Standard deviation of cosine similarity.
    """
    mean = np.mean(cos_sim, axis=0)
    std = np.std(cos_sim, axis=0)
    return mean, std


def evaluate(args):
    """Evaluate model.

    Args:
        args (dict): Configuration for evaluating trained model.

    """
    eval_stat = {
        "query_ids": [],
        "batch_query": [],
        "batch_pad_masks": [],
        "domain_label": [],
        "intent_label": [],
        "slots_label": [],
    }
    eval_start_time = time.strftime("%Y%m%d_%H-%M-%S")

    vocab_path = os.path.join(args["label_path"], "dict.txt")
    tokenizer = BasicTokenizer(vocab_path)
    label_processor = LabelProcessor(args["label_path"])
    evalset = MultiTaskDataset(
        args["evalset_path"],
        tokenizer,
        label_processor,
        args["seq_len"],
    )
    eval_loader = DataLoader(
        evalset,
        batch_size=args["batch_size"],
        num_workers=4,
        shuffle=False,
        collate_fn=lambda x: collate_func(x, args["seq_len"]),
    )

    print("load dataset")
    for _, batch_data in tqdm(
        enumerate(eval_loader),
        total=int(len(eval_loader.dataset) / args["batch_size"]) + 1,
    ):
        batch_query, batch_ids, batch_pad_masks = (
            batch_data["batch_query"],
            batch_data["batch_ids"],
            batch_data["batch_pad_masks"],
        )
        batch_domain_labs, batch_intent_labs, batch_ner_labs = (
            batch_data["batch_domain_labs"],
            batch_data["batch_intent_labs"],
            batch_data["batch_ner_labs"],
        )

        cur_domain_label = [
            label_processor.index_2_domain[i]
            for i in batch_domain_labs.tolist()
        ]
        cur_intent_label = [
            label_processor.index_2_intent[i]
            for i in batch_intent_labs.tolist()
        ]
        cur_slots_label = get_batch_ner_label(
            batch_ner_labs.tolist(),
            batch_pad_masks.sum(dim=1).tolist(),
            label_processor.index_2_slots,
        )

        eval_stat["query_ids"].extend(batch_ids)
        eval_stat["batch_query"].extend(batch_query)
        eval_stat["batch_pad_masks"].extend(batch_pad_masks)
        eval_stat["domain_label"].extend(cur_domain_label)
        eval_stat["intent_label"].extend(cur_intent_label)
        eval_stat["slots_label"].extend(cur_slots_label)

    print("finished loading dataset")

    true_samples = transfer_batch_sample(
        eval_stat["batch_query"],
        eval_stat["domain_label"],
        eval_stat["intent_label"],
        eval_stat["slots_label"],
    )
    original_pred_samples = []
    quantized_pred_samples = []

    original_predictor = OnnxPredictor(
        args, label_processor, args["original_onnx_file"]
    )
    quantized_predictor = OnnxPredictor(
        args, label_processor, args["quantized_onnx_file"]
    )

    seq_len = int(args["seq_len"])
    embedding = load_embedding(
        os.path.join(args["layers_path"], "embedding.float.weight.bin"),
    )

    domain_cos_sim = np.array([])
    intent_cos_sim = np.array([])
    slot_cos_sim = np.array([])
    slot_label_num = label_processor.slots_label_size

    for i, ids in tqdm(
        enumerate(eval_stat["query_ids"]), total=len(eval_stat["query_ids"])
    ):
        cur_query_ids = ids.numpy().tolist()
        cur_query = [eval_stat["batch_query"][i]]
        valid_length = eval_stat["batch_pad_masks"][i].sum(dim=0).tolist()

        input_emb = get_pad_embedding(embedding, cur_query_ids, seq_len)
        input_emb = np.array(input_emb, dtype=np.float32)  # [W, C]
        input_emb = input_emb.transpose(1, 0)  # [C, W]
        input_emb = np.expand_dims(input_emb, axis=0)  # [1, C, W]
        input_emb = np.expand_dims(input_emb, axis=2)  # [1, C, 1, W]

        (
            original_domain_out,
            original_intent_out,
            original_slots_out,
            original_preds,
        ) = original_predictor.predict(input_emb, cur_query, valid_length)
        (
            quantized_domain_out,
            quantized_intent_out,
            quantized_slots_out,
            quantized_preds,
        ) = quantized_predictor.predict(input_emb, cur_query, valid_length)

        original_pred_samples.extend(original_preds)
        quantized_pred_samples.extend(quantized_preds)

        cur_domain_sim = get_cos_sim(
            original_domain_out, quantized_domain_out
        ).reshape(1, 1)
        cur_intent_sim = get_cos_sim(
            original_intent_out, quantized_intent_out
        ).reshape(1, 1)
        cur_slot_sim = get_cos_sim(
            original_slots_out.reshape(-1, slot_label_num),
            quantized_slots_out.reshape(-1, slot_label_num),
        ).reshape(1, -1)

        if i == 0:
            domain_cos_sim = cur_domain_sim
            intent_cos_sim = cur_intent_sim
            slot_cos_sim = cur_slot_sim
        else:
            domain_cos_sim = np.concatenate(
                (domain_cos_sim, cur_domain_sim), axis=0
            )
            intent_cos_sim = np.concatenate(
                (intent_cos_sim, cur_intent_sim), axis=0
            )
            slot_cos_sim = np.concatenate((slot_cos_sim, cur_slot_sim), axis=0)

    (
        original_miss_recall,
        original_false_recall,
        original_other_diffs,
    ) = get_diff_samples(true_samples, original_pred_samples)
    (
        quant_miss_recall,
        quant_false_recall,
        quant_other_diffs,
    ) = get_diff_samples(true_samples, quantized_pred_samples)

    original_diff_num = (
        len(original_miss_recall)
        + len(original_false_recall)
        + len(original_other_diffs)
    )
    quantized_diff_num = (
        len(quant_miss_recall)
        + len(quant_false_recall)
        + len(quant_other_diffs)
    )

    original_acc = (
        (evalset.__len__() - original_diff_num) / evalset.__len__() * 100
    )
    quantized_acc = (
        (evalset.__len__() - quantized_diff_num) / evalset.__len__() * 100
    )

    domain_mean, domain_std = get_stat_info(domain_cos_sim)
    intent_mean, intent_std = get_stat_info(intent_cos_sim)
    slot_mean, slot_std = get_stat_info(slot_cos_sim)

    stat_dic = {
        "eval_start_time": eval_start_time,
        "evalset_path": args["evalset_path"],
        "original_onnx_file": args["original_onnx_file"],
        "quantized_onnx_file": args["quantized_onnx_file"],
        "total_samples_num": str(evalset.__len__()),
        "original_match_num": str(evalset.__len__() - original_diff_num),
        "original_diff_num": str(original_diff_num),
        "original_match_ratio(accuracy)": "%.4f%%" % original_acc,
        "quantized_match_num": str(evalset.__len__() - quantized_diff_num),
        "quantized_diff_num": str(quantized_diff_num),
        "quantized_match_ratio(accuracy)": "%.4f%%" % quantized_acc,
        "acc_diff": "%.4f%%" % (quantized_acc - original_acc),
        "acc_loss": "%.4f%%"
        % ((original_acc - quantized_acc) / original_acc * 100)
        if original_acc > 0
        else "NaN",
        "domain_cos_sim_mean": str(domain_mean[0]),
        "domain_cos_sim_std": str(domain_std[0]),
        "intent_cos_sim_mean": str(intent_mean[0]),
        "intent_cos_sim_std": str(intent_std[0]),
        "slot_cos_sim_mean": str(slot_mean.mean()),
        "slot_cos_sim_std": str(slot_std.mean()),
    }

    export_eval_report(
        args["output_path"],
        stat_dic,
    )


if __name__ == "__main__":
    if len(sys.argv) == 2:
        args = parse_args()
        evaluate(args)
    else:
        print(
            "Usage: python deploy/check_model_accuracy.py configs/tcn_ptq.py"
        )
