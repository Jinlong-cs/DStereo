# Copyright (c) Horizon Robotics. All rights reserved.
import json
import os
from collections import defaultdict
from typing import Dict, List

__all__ = [
    "slots_2_ner_label",
    "check_legality_of_ner_label",
    "get_batch_ner_label",
    "transfer_batch_sample",
    "get_diff_samples",
    "load_norm_dict",
    "export_eval_report",
]


def slots_2_ner_label(query, slots_label):
    """Transfer slots annotation into ner label list.

    Args:
        query: query str
        slots: list of slot annotaion

    Returns:
        ner_labels: list of ner label

    Example:
        query = "空调开到23度"
        slots = [{'raw': '空调', 'start': '0', 'end': '2', 'type': 'target'},
                {'raw': '23', 'start': '4', 'end': '6', 'type': 'value'}]
        ner_labels = get_ner_label(query, slots)
        #['B-target', 'E-target', 'O', 'O', 'B-value', 'E-value', 'O']

    """
    ner_labels = ["O" for i in range(len(query))]
    for i in range(len(slots_label)):
        cur_start = int(slots_label[i]["start"])
        cur_end = int(slots_label[i]["end"])
        cur_slice = slots_label[i]["raw"]
        cur_type = slots_label[i]["type"]
        if cur_slice != query[cur_start:cur_end]:
            raise UserWarning(
                "invalid ner label in this sample:\n \
                 query: {}\nslots: {}".format(
                    query, json.dumps(slots_label, ensure_ascii=False)
                )
            )
        if cur_end == cur_start + 1:
            ner_labels[cur_start] = "S-" + cur_type
        else:
            ner_labels[cur_start] = "B-" + cur_type
            ner_labels[cur_end - 1] = "E-" + cur_type
            ner_labels[cur_start + 1 : cur_end - 1] = [
                "I-" + cur_type for i in range(cur_end - cur_start - 2)
            ]
    return ner_labels


def check_legality_of_ner_label(query, slots_label):
    """Check whether slots label is legal and valid.

    Args:
        query: query str
        slots: list of slot annotaion

    Returns:
        True(legal) or False(illegal)

    Example:
        query = "空调开到23度"
        slots = [{'raw': '空调', 'start': '0', 'end': '2', 'type': 'target'},
                {'raw': '23', 'start': '4', 'end': '6', 'type': 'value'}]

    """
    for i in range(len(slots_label)):
        cur_start = int(slots_label[i]["start"])
        cur_end = int(slots_label[i]["end"])
        cur_slice = slots_label[i]["raw"]
        if (
            (cur_start > len(query) - 1)
            or cur_start < 0
            or cur_end > len(query)
            or (cur_end - 1 < cur_start)
            or cur_slice != query[cur_start:cur_end]
        ):
            return False
    return True


def get_batch_ner_label(batch_label_id, valid_lengths, index_2_ner):
    """Get batch ner label.

    Args:
        batch_label_id: list(N) of list(max_seq_length)
        valid_lengths: list(N)
        index_2_ner: dict

    Returns:
        batch_ner_label: list(N) of list(unfixed length depend on valid length)

    Example:
        batch_label_id = [[0, 1, 0], [1, 0, 2]]
        valid_lengths = [2, 3]
        index_2_ner = {0: "tag0", 1: "tag1", 2: "tag2"}
        print(get_batch_ner_label(batch_label_id, valid_lengths, index_2_ner))
        #[['tag0', 'tag1'], ['tag1', 'tag0', 'tag2']]

    """
    batch_valid_id = [
        batch_label_id[i][: valid_lengths[i]]
        for i in range(len(batch_label_id))
    ]
    batch_ner_label = [
        [index_2_ner[item] for item in seq] for seq in batch_valid_id
    ]
    return batch_ner_label


def transfer_batch_sample(
    batch_query, batch_domain_labels, batch_intent_labels, batch_ner_labels
):
    """Transfer separate query and labels batch data into batch samples.

    Args:
        batch_query: list
        batch_domain_labels: list
        batch_intent_labels: list
        batch_ner_labels: list

    Returns:
        batch_sample: list

    Example:
        batch_query = ['打开车窗', '打开车窗窗', '开一下空调']
        batch_domain_labels = ['window', 'window', 'window']
        batch_intent_labels = ['turn_on', 'turn_on', 'turn_on']
        batch_ner_labels = [['O', 'O', 'B-target', 'E-target'],
                            ['O', 'O', 'B-target', 'E-target', 'E-target'],
                            ['O', 'O', 'O', 'B-target', 'E-target']]
        transfer_batch_sample( batch_query, batch_domain_labels,
                               batch_intent_labels, batch_ner_labels)

    """
    assert (
        len(batch_query)
        == len(batch_domain_labels)
        == len(batch_intent_labels)
        == len(batch_ner_labels)
    )
    batch_sample = []
    for i in range(len(batch_query)):
        cur = {
            "query": batch_query[i],
            "domain": batch_domain_labels[i],
            "intent": batch_intent_labels[i],
            "slots": [
                {
                    "raw": batch_query[i][item[1] : item[2] + 1],
                    "type": item[0],
                    "start": item[1],
                    "end": item[2] + 1,
                }
                for item in get_entities(batch_ner_labels[i])
            ],
        }
        batch_sample.append(cur)
    return batch_sample


def get_diff_samples(batch_sample_true, batch_sample_pred):
    """Get samples which are predicted different from ground truth by the model.

    Different samples consists of miss recall and false recall samples:
    False recall samples: for domain and intent ,model both predicts
        non-other labels, and domain_pred or intent_pred is wrong.
    Miss recall samples: domain or intent true label != "other",
        but model predicts wrong label;
    Other diff samples: all other different samples

    Args:
        batch_sample_true: list (generate from transfer_batch_sample() )
        batch_sample_pred: list (generate from transfer_batch_sample() )

    Returns:
        list: list of miss recall samples.
        list: list of false recall samples.
        list: list of other diffrent samples.

    """
    assert len(batch_sample_true) == len(batch_sample_pred)
    false_recall_samples = []
    miss_recall_samples = []
    other_diff_samples = []
    for i in range(len(batch_sample_true)):
        assert batch_sample_true[i]["query"] == batch_sample_pred[i]["query"]
        query = batch_sample_true[i]["query"]
        if any(
            [
                batch_sample_true[i][k] != batch_sample_pred[i][k]
                for k in ["domain", "intent", "slots"]
            ]
        ):
            cur = {
                "query": query,
                "true_label": {
                    "domain": batch_sample_true[i]["domain"],
                    "intent": batch_sample_true[i]["intent"],
                    "slots": batch_sample_true[i]["slots"],
                },
                "pred_label": {
                    "domain": batch_sample_pred[i]["domain"],
                    "intent": batch_sample_pred[i]["intent"],
                    "slots": batch_sample_pred[i]["slots"],
                },
            }
            if all(
                [
                    batch_sample_pred[i][k] != "other"
                    for k in ["domain", "intent"]
                ]
            ) and any(
                [
                    batch_sample_true[i][k] != batch_sample_pred[i][k]
                    for k in ["domain", "intent"]
                ]
            ):
                false_recall_samples.append(cur)
            elif any(
                [
                    batch_sample_true[i][k] != "other"
                    and batch_sample_true[i][k] != batch_sample_pred[i][k]
                    for k in ["domain", "intent"]
                ]
            ):
                miss_recall_samples.append(cur)
            else:
                other_diff_samples.append(cur)
    return miss_recall_samples, false_recall_samples, other_diff_samples


def load_norm_dict(norm_dict_path):
    """Load norm dict file (format as nlu project).

    for example:https://gitlab.hobot.cc/ptd/algorithm/nlp/nlu/-/tree/master/
                       nlu/data/standard/nn/norm
    Args:
        norm_dict_path: str
    Returns:
        norm_dict: defaultdict(lambda: defaultdict(str))
    """
    types_all = os.listdir(norm_dict_path)
    norm_dict = defaultdict(lambda: defaultdict(str))
    for type in types_all:
        file_all = os.listdir(os.path.join(norm_dict_path, type))
        for file in file_all:
            with open(
                os.path.join(norm_dict_path, type, file),
                "r",
                encoding="utf-8",
            ) as f:
                all_lines = f.readlines()
            for line in all_lines:
                line = line.strip()
                norm = line.split(":")[0]
                norm_dict[type][norm] = norm
                if ":" in line:
                    raw_values = line.split(":")[1].split(",")
                    for raw in raw_values:
                        norm_dict[type][raw] = norm
    return norm_dict


def export_eval_report(save_path: str, stat_dic: Dict):
    """Export model evaluation report.

    Args:
        save_path: Path to save the report.
        stat_dic: Python dict of
            statistical results at evaluation stage.

    """
    with open(save_path, "w", encoding="utf-8") as f:
        for k in stat_dic:
            f.write("**" + k + "**" + "\n")
            if k in [
                "miss_recall_samples",
                "false_recall_samples",
                "other_diff_samples",
                "protocol_false_recall_samples",
            ]:
                for item in stat_dic[k]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                f.write(stat_dic[k] + "\n")


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Check if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.

    """
    chunk_start = False

    if tag == "B":
        chunk_start = True
    if tag == "S":
        chunk_start = True

    if prev_tag == "E" and tag == "E":
        chunk_start = True
    if prev_tag == "E" and tag == "I":
        chunk_start = True
    if prev_tag == "S" and tag == "E":
        chunk_start = True
    if prev_tag == "S" and tag == "I":
        chunk_start = True
    if prev_tag == "O" and tag == "E":
        chunk_start = True
    if prev_tag == "O" and tag == "I":
        chunk_start = True

    if tag != "O" and tag != "." and prev_type != type_:
        chunk_start = True

    return chunk_start


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Check if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.

    """
    chunk_end = False

    if prev_tag == "E":
        chunk_end = True
    if prev_tag == "S":
        chunk_end = True

    if prev_tag == "B" and tag == "B":
        chunk_end = True
    if prev_tag == "B" and tag == "S":
        chunk_end = True
    if prev_tag == "B" and tag == "O":
        chunk_end = True
    if prev_tag == "I" and tag == "B":
        chunk_end = True
    if prev_tag == "I" and tag == "S":
        chunk_end = True
    if prev_tag == "I" and tag == "O":
        chunk_end = True

    if prev_tag != "O" and prev_tag != "." and prev_type != type_:
        chunk_end = True

    return chunk_end


def get_entities(seq: List, suffix=False) -> List:
    """Get entities from sequence.

    Args:
        seq: sequence of labels.

    Returns:
        list of (chunk_type, chunk_start, chunk_end).

    Example:
            from seqeval.metrics.sequence_labeling import get_entities
            seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
            get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]

    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ["O"]]

    prev_tag = "O"
    prev_type = ""
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ["O"]):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split("-")[0]
        else:
            tag = chunk[0]
            type_ = chunk.split("-")[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks
