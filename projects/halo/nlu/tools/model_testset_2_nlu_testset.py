"""模型测试集转换为nlu引擎测试集格式."""

__author__ = "ruosong.li@horizon.ai"
__date__ = "2022/11/14"
__copyright__ = "Copyright (C) 2022 Horizon Robotics, Inc."

import argparse
import json
import os
import sys
from collections import defaultdict

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_PATH, "../../../../"))
from hat.core.nlu.nlu_protocol_eval import Normalizer  # noqa: E402


def parse_args():
    """输入参数解析."""
    parser = argparse.ArgumentParser(
        description="NLU transfer model testset to nlu engine format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        default="",
        help="input path of model testset",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="",
        help="output path of nlu testset",
    )
    parser.add_argument(
        "-n",
        "--normalization",
        type=bool,
        default=False,
        help="whether to do slot normalization",
    )
    args = parser.parse_args()
    return args


class FormatConvertor:
    """模型测试集格式转换为nlu引擎测试集格式。
    样例:
    转换前格式:
    {"query": "把右前车窗打开",
     "domain": "window",
     "intent": "turn_on",
     "slots": [{"type": "target", "raw": "车窗", "start": 3, "end": 5},
               {"type": "position", "raw": "右前", "start": 1, "end": 3}],
    }

    转换后格式（-n false）:
    {"query": "把右前车窗打开",
     "result": [{"domain": "window",
                 "intent": "turn_on",
                 "slots": {"target": [{"normalized": "车窗", "raw": "车窗", "start": 3, "end": 5}],   # noqa: E501
                           "position": [{"normalized": "右前", "raw": "右前", "start": 1, "end": 3}]} # noqa: E501
                          }
               ]
    }

    转换后格式（-n true）:
    {"query": "把右前车窗打开",
     "result": [{"domain": "window",
                 "intent": "turn_on",
                 "slots": {"target": [{"normalized": "车窗", "raw": "车窗", "start": 3, "end": 5}],   # noqa: E501
                           "position": [{"normalized": "副驾", "raw": "右前", "start": 1, "end": 3}]  # noqa: E501
                          }
                }
               ]
    }

    """

    def __init__(self, normalization=False):
        self.normalization = normalization
        self.normalizer = Normalizer() if self.normalization else None

    def transfer(self, input_dic):
        """transfer model format json to nlu format json."""
        output_dic = {"query": input_dic["query"]}
        result = {"domain": input_dic["domain"], "intent": input_dic["intent"]}
        slots = defaultdict(list)
        for slot in input_dic["slots"]:
            if self.normalization:
                slots[slot["type"]].append(
                    {
                        "normalized": self.normalizer.normalize(
                            slot["type"], slot["raw"]
                        ),
                        "raw": slot["raw"],
                        "start": slot["start"],
                        "end": slot["end"],
                    }
                )
            else:
                slots[slot["type"]].append(
                    {
                        "normalized": slot["raw"],
                        "raw": slot["raw"],
                        "start": slot["start"],
                        "end": slot["end"],
                    }
                )
        result["slots"] = slots
        output_dic["result"] = [result]
        return output_dic


if __name__ == "__main__":
    args = parse_args()
    current_work_dir = os.getcwd()
    input_file = os.path.join(current_work_dir, args.input_path)
    output_file = os.path.join(current_work_dir, args.output_path)
    norm_flag = args.normalization
    fc = FormatConvertor(norm_flag)
    if not os.path.exists(input_file):
        raise UserWarning(
            "Input path: {} not exist! Please check!".format(input_file)
        )
    with open(input_file, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    all_data = [json.loads(i) for i in all_lines]
    output_data = [fc.transfer(i) for i in all_data]
    with open(output_file, "w", encoding="utf-8") as f:
        for data in output_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
