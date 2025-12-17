# Copyright (c) Horizon Robotics. All rights reserved.
import json
import logging
import os
from collections import defaultdict
from typing import List, Optional

from torch.utils.data import Dataset

from hat.core.nlu.nlu_utils import (
    check_legality_of_ner_label,
    slots_2_ner_label,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.filesystem import get_filesystem

__all__ = [
    "NluMultiTaskDataset",
    "NluLabelProcessor",
    "NluBasicTokenizer",
]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class NluMultiTaskDataset(Dataset):
    """Construct dataset for multi-task model."""

    def __init__(
        self,
        data_path: str,
        tokenizer: "BasicTokenizer",  # noqa
        label_processor: "LabelProcessor",  # noqa
        max_query_length: Optional[int] = 63,
        data_report_path: Optional[str] = "",
    ):
        """Init MultiTaskDataset class.

        Args:
            data_path: Path to data file.
            tokenizer: Tokenize query into token id list.
            label_processor: Process original label to
                corresponding label id.
            max_query_length: Maximum length of input query.
            data_report_path: report path of data

        """
        self.tokenizer = tokenizer
        self.label_processor = label_processor
        self.max_query_length = max_query_length
        self.data_report_path = data_report_path
        self.data_all = self._load_data(data_path)
        self.sample_weight_dict = defaultdict(
            self._get_default_weight,
            {
                "regex": 1,
                "句式采集": 2,
                "lixiang_regex": 5,
                "lixiang_label": 5,
                "negative": 2,
            },
        )

    def _get_default_weight(self):
        return 1

    def _load_data(self, data_path):
        jfs = get_filesystem(data_path)
        with jfs.open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        data_all = [json.loads(item) for item in lines]
        valid_data = []
        data_stat_logs = []
        for i in range(len(data_all)):
            cur_data = data_all[i]
            if any(
                [
                    k not in cur_data
                    for k in ["query", "domain", "intent", "slots"]
                ]
            ):
                data_stat_logs.append(
                    "".join(
                        ["invalid data: missing key in line {} ,", "data: {} "]
                    ).format(
                        str(i + 1), json.dumps(cur_data, ensure_ascii=False)
                    )
                )
                continue
            elif (
                len(cur_data["query"]) > self.max_query_length
                or len(cur_data["query"]) < 2
            ):
                data_stat_logs.append(
                    "".join(
                        [
                            "invalid data: illegal query length ",
                            "in line {} , data: {}",
                        ]
                    ).format(
                        str(i + 1), json.dumps(cur_data, ensure_ascii=False)
                    )
                )
                continue
            elif (
                cur_data["domain"] not in self.label_processor.domain_2_index
                or cur_data["intent"]
                not in self.label_processor.intent_2_index
                or any(
                    [
                        slots["type"] not in self.label_processor.slots_types
                        for slots in cur_data["slots"]
                    ]
                )
            ):
                data_stat_logs.append(
                    "".join(
                        [
                            "invalid data: illegal label type ",
                            "in line {} , data: {} ",
                        ]
                    ).format(
                        str(i + 1), json.dumps(cur_data, ensure_ascii=False)
                    )
                )
                continue
            elif not check_legality_of_ner_label(
                cur_data["query"], cur_data["slots"]
            ):
                data_stat_logs.append(
                    "".join(
                        [
                            "invalid data: illegal slots label ",
                            "in line {} , data: {} ",
                        ]
                    ).format(
                        str(i + 1), json.dumps(cur_data, ensure_ascii=False)
                    )
                )
                continue
            else:
                valid_data.append(cur_data)
        total_stat = "".join(
            [
                "Data path total stats: \n",
                "data path:{}\n",
                "total data num:{}\n",
                "valid data num:{}",
            ]
        ).format(data_path, str(len(data_all)), str(len(valid_data)))
        logging.info(total_stat)
        data_stat_logs.append(total_stat)
        if self.data_report_path:
            if not os.path.exists(os.path.dirname(self.data_report_path)):
                os.mkdir(os.path.dirname(self.data_report_path))
            with open(self.data_report_path, "w", encoding="utf-8") as f:
                for line in data_stat_logs:
                    f.write(line + "\n")
        return valid_data

    def __len__(self):
        """Get numbers of samples.

        Returns:
            int: Number of samples.

        """
        return len(self.data_all)

    def __getitem__(self, index):
        """Get sample data.

        Returns:
            dict: Python dict of sample data, including query and label.

        """
        query = self.data_all[index]["query"]
        domain = self.data_all[index]["domain"]
        intent = self.data_all[index]["intent"]
        ner = slots_2_ner_label(query, self.data_all[index]["slots"])
        sample_weight = (
            self.sample_weight_dict[self.data_all[index]["source"]]
            if self.data_all[index].get("source") is not None
            else 1
        )
        sample = {
            "query": query,
            "query_id": self.tokenizer.tokenize(
                ["<dom>", "<int>"] + list(query)
            ),
            "domain": self.label_processor.domain_2_index[domain],
            "intent": self.label_processor.intent_2_index[intent],
            "ner": [self.label_processor.slots_2_index[lab] for lab in ner],
            "sample_weight": int(sample_weight),
        }

        return sample


@OBJECT_REGISTRY.register
class NluLabelProcessor:
    """Processor of multi-task labels.

    Attributes:
        domain_2_index: map dict
        index_2_domain: map dict
        intent_2_index: map dict
        index_2_intent: map dict
        slots_2_index: map dict
        index_2_slots: map dict
        domain_label_size: int
        intent_label_size: int
        slots_label_size: int

    """

    def __init__(self, label_path: str):
        """Init LabelProcessor class.

        Args:
            label_path: Path to label folder.

        """
        domain_path = os.path.join(label_path, "domain.txt")
        intent_path = os.path.join(label_path, "intent.txt")
        slots_path = os.path.join(label_path, "slots.txt")
        self.domain_2_index, self.index_2_domain = self._load_label_file(
            domain_path
        )
        self.intent_2_index, self.index_2_intent = self._load_label_file(
            intent_path
        )
        self.slots_2_index, self.index_2_slots = self._load_label_file(
            slots_path
        )
        self.slots_types = list(
            {k.split("-")[1] for k in self.slots_2_index if "-" in k}
        )
        self.domain_label_size = len(self.domain_2_index)
        self.intent_label_size = len(self.intent_2_index)
        self.slots_label_size = len(self.slots_2_index)

    def _load_label_file(self, file_path):
        jfs = get_filesystem(file_path)
        with jfs.open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        labels = [item.replace("\n", "") for item in lines]
        if len(labels) != len(list(set(labels))):  # 判断是否有有重复的label
            error_info = "Same label found in {} , please check!".format(
                file_path
            )
            raise UserWarning(error_info)
        if not labels:
            error_info = "{} is empty, please check!".format(file_path)
            raise UserWarning(error_info)
        label_2_index, index_2_label = {}, {}
        for i in range(len(labels)):
            label_2_index[labels[i]] = i
            index_2_label[i] = labels[i]
        return label_2_index, index_2_label


@OBJECT_REGISTRY.register
class NluBasicTokenizer:
    """Basic tokenizer class.

    Methods:
        tokenize():  transform word list to token-index list
        detokenize(): transform token-index list to word list

    """

    def __init__(self, vocab_path: str):
        """Init BasicTokenizer class.

        Args:
            vocab_path: Path to vocabulary file.

        """
        self.vocab = self._load_vocab(vocab_path)
        self.vocab_size = len(self.vocab)
        self.v_2_index = {self.vocab[i]: i for i in range(self.vocab_size)}
        self.index_2_v = {i: self.vocab[i] for i in range(self.vocab_size)}

    def _load_vocab(self, vocab_path):
        """Load vocab file , treat each line as one word."""
        jfs = get_filesystem(vocab_path)
        with jfs.open(vocab_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            vocab = [item.replace("\n", "") for item in lines]
        return vocab

    def tokenize(self, query_list: List):
        """Transform query list into token index list.

        Args:
            query_list: List of query tokens.

        Returns:
            List: Python list of token index.

        """
        unk_inx = self.v_2_index["<unk>"]
        token_list = [self.v_2_index.get(c, unk_inx) for c in query_list]
        return token_list

    def detokenize(self, token_list: List):
        """Transform token index into original query.

        Args:
            token_list: List of token idx.

        Returns:
            List: Python list of original query token.

        """
        return [self.index_2_v[t] for t in token_list]
