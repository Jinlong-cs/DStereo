# Copyright (c) Horizon Robotics. All rights reserved.

import json
import os
from collections import defaultdict

import requests

from hat.core.nlu.nlu_utils import load_norm_dict

protocol_url = (
    "http://nlu.hobot.cc/v1/nlu/protocol/protocol-json?project=standard_halo5"
)

__all__ = [
    "Normalizer",
    "ProtocolEvaluator",
]


class Normalizer:
    """Normalizer."""

    def __init__(
        self,
        norm_dict_path=None,
    ):
        """Initialize normalizer."""
        norm_dict_path = (
            os.getenv("NLU_NORM_FILE_PATH")
            if not norm_dict_path
            else norm_dict_path
        )
        self._load_norm_dict(norm_dict_path)

    def _load_norm_dict(self, norm_dict_path):
        """Load norm dict file in nlu project."""
        self._norm_dict = load_norm_dict(norm_dict_path)

    def normalize(self, type, raw_value):
        """Normalize method,same strategy as nlu engine.

        Args:
            type:  str,slot_type
            raw_value: str,raw_slot_value
        Returns:
            norm_value: str,norm_slot_value
        """
        norm_value = self._norm_dict[type][raw_value]
        return norm_value if norm_value else raw_value


class ProtocolEvaluator:
    """ProtocolEvaluator."""

    def __init__(self, normalizer, protocol_path=protocol_url) -> None:
        """Initialize protocol evaluator."""
        self.protocol_path = protocol_path
        self._load_protocol(protocol_path)
        self.normalizer = normalizer

    def _load_protocol(self, protocol_path):
        """Load protocol."""
        res = json.loads(requests.get(protocol_path).text)
        if res["code"] != "200":
            raise UserWarning(
                "Failed to load protocol from {}".format(self.protocol_path)
            )
        protocol_raw = []
        for k in res["data"]:
            protocol_raw.extend(res["data"][k])
        protocol_all = [
            defaultdict(str, protocol) for protocol in protocol_raw
        ]
        slots_map = defaultdict(list)

        for protocol in protocol_all:
            cur_key = protocol["domain"] + "__" + protocol["intent"]
            all_slots = {k: v.split(",") for k, v in protocol["slots"].items()}
            required_slots = [
                item.split("|") for item in protocol["required_slots"]
            ]
            cur_func = {
                "all_slots": all_slots,
                "required_slots": required_slots,
                "third_category": protocol["third_category"],
            }
            slots_map[cur_key].append(cur_func)

        self._slots_map = slots_map

    def is_legal(self, pred_result):
        """Check whether input_result is legal(defined by protocol.json).

        Args:
            pred_result = {"domain":xxx,
                           "intent":xxx,
                           "slots":[{"type":xxx,"raw":xxx}]}
        Returns:
            bool.
        """
        cur_key = pred_result["domain"] + "__" + pred_result["intent"]
        # check if pred_slot is empty
        if not pred_result["slots"]:
            return False
        # check domain and intent
        if cur_key not in self._slots_map:
            return False
        # check if pred_result match any func protocol
        for func_protocol in self._slots_map[cur_key]:
            match_flag = True
            # check all pred slot is legal (ignore "value" and "gear" slot)
            for pred_slot in pred_result["slots"]:
                slot_type = pred_slot["type"]
                raw_value = pred_slot["raw"]
                norm_value = self.normalizer.normalize(slot_type, raw_value)
                if (slot_type not in func_protocol["all_slots"]) or (
                    slot_type not in ["value", "gear"]
                    and norm_value not in func_protocol["all_slots"][slot_type]
                ):
                    match_flag = False
                    break
            # check pred result contain all required slots
            for required_slots in func_protocol["required_slots"]:
                if not any(
                    [
                        (
                            slot_type in ["value", "gear"]
                            and any(
                                [
                                    pred_slot["type"] == slot_type
                                    for pred_slot in pred_result["slots"]
                                ]
                            )
                        )
                        or any(
                            [
                                pred_slot["type"] == slot_type
                                and self.normalizer.normalize(
                                    slot_type, pred_slot["raw"]
                                )
                                in func_protocol["all_slots"][slot_type]
                                for pred_slot in pred_result["slots"]
                            ]
                        )
                        for slot_type in required_slots
                    ]
                ):
                    match_flag = False
                    break
            if match_flag:
                return True
        return False


if __name__ == "__main__":
    normalizer = Normalizer()
    protocol_evaluator = ProtocolEvaluator(normalizer)
    # false
    print(
        protocol_evaluator.is_legal(
            {
                "domain": "vehicle_other",
                "intent": "turn_on",
                "slots": [{"raw": "xx记录仪", "type": "target"}],
            }
        )
    )
    # true
    print(
        protocol_evaluator.is_legal(
            {
                "domain": "setting",
                "intent": "turn_on",
                "slots": [{"raw": "行车记录仪", "type": "target"}],
            }
        )
    )
    # false
    print(
        protocol_evaluator.is_legal(
            {
                "domain": "system",
                "intent": "turn_on",
                "slots": [{"raw": "一个新app", "type": "app"}],
            }
        )
    )
    # true
    print(
        protocol_evaluator.is_legal(
            {
                "domain": "system",
                "intent": "turn_on",
                "slots": [{"raw": "酷我音乐", "type": "app"}],
            }
        )
    )
    # false
    print(
        protocol_evaluator.is_legal(
            {
                "domain": "other",
                "intent": "other",
                "slots": [{"raw": "行车记录仪", "type": "target"}],
            }
        )
    )
    # false
    print(
        protocol_evaluator.is_legal(
            {"domain": "system", "intent": "turn_on", "slots": []}
        )
    )
