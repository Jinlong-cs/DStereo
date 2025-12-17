"""基于转换后的标准json数据生成nlu数据集."""

__author__ = "ruosong.li@horizon.ai"
__date__ = "2022/06/07"
__copyright__ = "Copyright (C) 2022 Horizon Robotics, Inc."

import argparse
import itertools
import json
import logging
import math
import os
import re
import sys
from bisect import bisect
from collections import defaultdict
from random import choice, random, shuffle

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_PATH, "../../../../"))
from hat.core.nlu.nlu_utils import load_norm_dict  # noqa: E402


def parse_args():
    """输入参数解析."""
    parser = argparse.ArgumentParser(
        description="NLU generate dataset from std json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f",
        "--file_path",
        type=str,
        default="./files/nlu语义_v1.0_训练数据采集_vehicle.csv_std_format.json",
        help="file path of std format json",
    )
    parser.add_argument(
        "-o",
        "--is_open_domain",
        type=bool,
        default=False,
        help="open domain flag",
    )
    parser.add_argument(
        "-pp",
        "--pos_path",
        type=str,
        default="./files/nlu_v1.0.posset",
        help="file path to save posset",
    )
    parser.add_argument(
        "-np",
        "--neg_path",
        type=str,
        default="./files/nlu_v1.0.negset",
        help="file path to save negset",
    )
    parser.add_argument(
        "-lp",
        "--log_path",
        type=str,
        default="./files/data_sampler.log",
        help="file path to save log",
    )
    args = parser.parse_args()
    return args


def int_2_chn(i):
    """一百以内整数转成中文： 0 -> 零 , 15->十五 , 100 -> 一百."""
    assert isinstance(i, int) and i >= 0 and i <= 100
    base_map = ["十", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
    if i == 0:
        return "零"
    if i == 100:
        return "一百"
    if i < 10:
        return base_map[i]
    res = []
    p1 = i // 10
    if p1 == 1:
        res.append("十")
    else:
        res.append(base_map[p1] + "十")
    p2 = i % 10
    if p2 != 0:
        res.append(base_map[p2])
    return "".join(res)


def value_percent(n=10):
    """百分数值采样: 百分之三十，百分之30，30%."""
    nums_all = list(range(101))
    res = []
    for _ in range(n):
        cur_num = choice(nums_all)
        # 百分之  或  %
        flag_pat = random()
        # 5的倍数  或 任意随机数
        flag_num = random()
        # 百分之 三十  或  30
        flag_chn = random()
        if cur_num > 5 and flag_num > 0.1:
            cur_num -= cur_num % 5
        if flag_pat < 0.5:
            res.append(str(cur_num) + "%")
        else:
            if flag_chn < 0.8:
                res.append("百分之" + int_2_chn(cur_num))
            else:
                res.append("百分之" + str(cur_num))
    return res


def value_frac(n=10):
    """分数数值采样: 三分之一 ， 1/3."""
    res = []
    for _ in range(n):
        # 三分之一  或  1/3
        flag_pat = random()
        down = choice(list(range(2, 11)))
        up = choice(list(range(1, down)))
        if flag_pat > 0.5:
            res.append(str(up) + "/" + str(down))
        else:
            res.append(int_2_chn(down) + "分之" + int_2_chn(up))
    return res


def value_int(n=10):
    """百以内整数采样 : 15 或 十五."""
    res = []
    for _ in range(n):
        # 数字或中文
        flag_pat = random()
        # 5的倍数  或 任意随机数
        flag_num = random()
        val = choice(list(range(0, 101)))
        if val > 5 and flag_num > 0.1:
            val -= val % 5
        if flag_pat > 0.5:
            res.append(str(val))
        else:
            res.append(int_2_chn(val))
    return res


def value_decimai(n=10):
    """小数数值采样."""
    int_part = list(range(0, 50))
    decimai_part = list(range(1, 10))
    res = []

    for _ in range(n):
        cur_int = choice(int_part)
        cur_dec = choice(decimai_part)
        if random() > 0.5:
            res.append(str(cur_int) + "." + str(cur_dec))
        else:
            res.append(int_2_chn(cur_int) + "点" + int_2_chn(cur_dec))
    return res


def value_time(n=10):
    """时间数值采样."""
    candi_spe = [
        "半小时",
        "半个小时",
        "一小时",
        "一个小时",
        "1小时",
        "2小时",
        "两小时",
        "两个小时",
        "俩小时",
        "一个半小时",
    ]
    res = []
    for _ in range(n):
        flag_num = random()
        flag_pat = random()
        flag_chn = random()
        if flag_pat > 0.5:
            res.append(choice(candi_spe))
        else:
            val = choice(list(range(1, 61)))
            if val > 5 and flag_num > 0.1:
                val -= val % 5
            if flag_chn > 0.5:
                s = str(val)
            else:
                s = int_2_chn(val)
            res.append(s + "分钟")
    return res


def value_radio(n=10):
    """电台数值采样."""
    int_part = list(range(88, 109))
    decimai_part = list(range(1, 10))
    res = []

    for _ in range(n):
        cur_int = choice(int_part)
        cur_dec = choice(decimai_part)
        flag = random()
        if cur_int > 100 or flag > 0.7:
            res.append(str(cur_int) + "." + str(cur_dec))
        elif flag > 0.4:
            res.append(int_2_chn(cur_int) + "点" + int_2_chn(cur_dec))
        elif flag > 0.2:
            res.append(str(cur_int))
        else:
            res.append(int_2_chn(cur_int))
    return res


def get_sample_ratio(
    all_num,
    breakpoints=(1000, 10000, 50000, 100000),
    ratios=(1, 0.8, 0.5, 0.3, 0.2),
):
    """获取区间采样率."""
    i = bisect(breakpoints, all_num)
    return ratios[i]


class BaseSampler:
    """基于三级功能点的采样器."""

    def __init__(self, std_dic, is_open_domain=False):
        """采样器初始化."""
        self.sample_slot = {
            "value": ["百分数", "分数", "小数", "整数", "电台数值"],
            "time": ["时间数值"],
        }
        self.source_tag = "句式采集"
        self._load_std_dic(std_dic)
        self.is_open_domain = is_open_domain

    def _load_std_dic(self, std_dic):
        """加载功能点相关字段."""
        self.func = std_dic["三级功能点"]
        self.domain = std_dic["domain"]
        self.intent = std_dic["intent"]
        self.std_slot = std_dic["slot"]
        self.slot_value = std_dic["slot_value"]
        self.sent_pats = list(set(std_dic["sents_pattern"]))
        self.neg_sent_pats = list(set(std_dic["neg_sents_pattern"]))

    def _add_prefix_suffix_in_sent(self, sample_dict, positive=False):
        """数据增强：为句式样本随机添加前后缀."""
        all_candi_prefix = [
            "麻烦请给我",
            "麻烦帮我",
            "麻烦给我",
            "我要把",
            "请帮我",
            "让那个",
            "我希望",
            "就那个",
            "能帮我",
            "我想把",
            "能不能",
            "能给我",
            "我需要",
            "请给我",
            "方便",
            "帮我",
            "立马",
            "赶紧",
            "我要",
            "让你",
            "我想",
            "立即",
            "立刻",
            "可以",
            "帮忙",
            "要把",
            "就是",
            "想把",
            "麻烦",
            "劳烦",
            "给我",
            "马上",
            "那个",
            "迅速",
            "要",
            "让",
            "请",
            "给",
            "就",
            "帮",
            "想",
        ]
        all_candi_suffix = [
            "成不成",
            "可以吗",
            "好不好",
            "行不行",
            "可以吧",
            "可以么",
            "好吗",
            "成不",
            "行吧",
            "好么",
            "行吗",
            "行么",
            "成吧",
            "好不",
            "好吧",
            "行不",
            "哦",
            "吗",
            "吧",
            "嘞",
            "呀",
            "了",
            "呗",
            "呦",
            "喽",
        ]
        all_candi_common = ["哎呀", "诶", "嗯", "哎", "啊", "呃"]

        prefix_thres = 0.8
        suffix_thres = 0.8
        sent_str = sample_dict["query"]
        prefix = ""
        suffix = ""
        if random() > prefix_thres:
            prefix = choice(all_candi_prefix + all_candi_common)
            sent_str = prefix + sent_str
        if random() > suffix_thres:
            suffix = choice(all_candi_suffix + all_candi_common)
            sent_str = sent_str + suffix
        sample_dict["query"] = sent_str
        if positive:
            for i in range(len(sample_dict["slots"])):
                sample_dict["slots"][i]["start"] += len(prefix)
                sample_dict["slots"][i]["end"] += len(prefix)

    def _get_sample_values(self, slot, std_value, n=10):
        """获取基于采样函数的槽位值."""
        assert slot in self.sample_slot and std_value in self.sample_slot[slot]
        if std_value == "百分数":
            return value_percent(n)
        if std_value == "分数":
            return value_frac(n)
        if std_value == "小数":
            return value_decimai(n)
        if std_value == "整数":
            return value_int(n)
        if std_value == "时间数值":
            return value_time(n)
        if std_value == "电台数值":
            return value_radio(n)

    def _get_all_slot_values(self, slot):
        """获取给定槽位的全部候选值."""
        all_candi_slot_value = []
        if slot not in self.std_slot:
            assert "-" in slot
            return all_candi_slot_value
        slot_std_values = self.std_slot[slot]
        for std_value in slot_std_values:
            # 基于采样函数获取候选值
            if (
                slot in self.sample_slot
                and std_value in self.sample_slot[slot]
            ):
                all_candi_slot_value.extend(
                    self._get_sample_values(slot, std_value)
                )
            # 基于给定标注数据获取候选值
            else:
                if slot + "@" + std_value in self.slot_value:
                    all_candi_slot_value.extend(
                        self.slot_value[slot + "@" + std_value]
                    )
        return all_candi_slot_value

    def _get_slots_in_sent(self, sent_str):
        """解析句式中的 {xxx} 格式的槽位."""
        pat = re.compile(r"{[\w-]*}")
        slots_all = pat.findall(sent_str)
        return slots_all

    def _slot_value_resample(self, slot_values, resample_num):
        """槽位值重采样到指定数值."""
        assert len(slot_values) > 1 and resample_num > 0
        res = []
        while len(res) < resample_num:
            res.extend(list(slot_values))
        return res[:resample_num]

    def _get_sent_sample(
        self,
        sent_str,
        positive,
        no_slot_resample=20,
        open_domain_resample=6000,
    ):
        """基于单句句式生成样本.

        sent_str : str.
        positive : bool , 正样本->True,负样本->False.
        no_slot_resample: 无槽位样本的过采样比例.
        open_domain_resample： 开放域样本的采样数量.
        """
        slots_all = self._get_slots_in_sent(sent_str)

        # 不带槽位的单句样本
        if not slots_all:
            if positive:
                cur_sample = {
                    "domain": self.domain,
                    "intent": self.intent,
                    "query": sent_str,
                    "slots": [],
                    "source": self.source_tag,
                }
            else:
                cur_sample = {
                    "domain": "other",
                    "intent": "other",
                    "query": sent_str,
                    "slots": [],
                    "source": self.source_tag,
                }
            return [cur_sample for _ in range(no_slot_resample)]

        # 车控域
        if not self.is_open_domain:
            # 获取全部待替换槽位值
            cur_replace_lists = []
            for cur_slot in slots_all:
                slot_name = cur_slot[1:-1]
                cur_replace_lists.append(
                    [
                        (cur_slot, slot_value)
                        for slot_value in self._get_all_slot_values(slot_name)
                    ]
                )
            all_candi_replace = list(itertools.product(*cur_replace_lists))
        # 开放域
        else:
            # 整理槽位可选值
            all_slot_values = defaultdict(list)
            combo_cnt = 1
            for cur_slot in slots_all:
                # 每种槽位应至少有标注槽位值、词表值中的一种
                slot_name = cur_slot[1:-1]
                all_slot_values[slot_name].extend(
                    self._get_all_slot_values(slot_name)
                )
                if slot_name in open_domain_dict:
                    slot_values_from_dict = list(
                        open_domain_dict[slot_name].keys()
                    )
                    shuffle(slot_values_from_dict)
                    all_slot_values[slot_name].extend(slot_values_from_dict)
                combo_cnt *= len(all_slot_values[slot_name])
            samples_all = []
            # 槽值组合数量较小，采用与车控域一致的采样方式
            if combo_cnt < open_domain_resample:
                cur_replace_lists = []
                for cur_slot in slots_all:
                    slot_name = cur_slot[1:-1]
                    cur_replace_lists.append(
                        [
                            (cur_slot, slot_value)
                            for slot_value in all_slot_values[slot_name]
                        ]
                    )
                all_candi_replace = list(itertools.product(*cur_replace_lists))
            # 槽值组合较多，采用开放域采样方式
            else:
                for cur_slot in slots_all:
                    slot_name = cur_slot[1:-1]
                    all_slot_values[slot_name] = self._slot_value_resample(
                        tuple(all_slot_values[slot_name]), open_domain_resample
                    )
                all_candi_replace = []
                for i in range(open_domain_resample):
                    cur_replace = []
                    for cur_slot in slots_all:
                        slot_name = cur_slot[1:-1]
                        cur_replace.append(
                            (cur_slot, all_slot_values[slot_name][i])
                        )
                    all_candi_replace.append(tuple(cur_replace))

        samples_all = []
        # 基于每种替换方式逐个生成样本
        for cur_replace_all in all_candi_replace:
            if positive:
                cur_sample = {
                    "domain": self.domain,
                    "intent": self.intent,
                    "slots": [],
                    "source": self.source_tag,
                }
            else:
                cur_sample = {
                    "domain": "other",
                    "intent": "other",
                    "slots": [],
                    "source": self.source_tag,
                }
            cur_sent_str = sent_str
            for slot_name, slot_value in cur_replace_all:
                start_idx = cur_sent_str.index(slot_name)
                cur_sent_str = cur_sent_str.replace(slot_name, slot_value)
                if positive:
                    # 对于含#标记的槽位，槽位类型保留#前部分，例:end_location#place->end_location
                    slot_type = (
                        slot_name[1:-1].split("-")[0]
                        if "-" in slot_name
                        else slot_name[1:-1]
                    )
                    cur_sample["slots"].append(
                        {
                            "raw": slot_value,
                            "start": start_idx,
                            "end": start_idx + len(slot_value),
                            "type": slot_type,
                        }
                    )
            cur_sample["query"] = cur_sent_str
            samples_all.append(cur_sample)

        # 车控领域采样 (结合非槽位字符数量以及槽位数量进行重采样。)
        if not self.is_open_domain:
            if positive:
                sent_without_slot = re.sub(re.compile(r"{\w*}"), "", sent_str)
                sample_ratio = get_sample_ratio(
                    len(sent_without_slot), (5, 10, 15, 20), (20, 15, 8, 5, 3)
                )
                sample_nums = sample_ratio * len(slots_all)
            else:
                sample_nums = 10 * len(slots_all)
            shuffle(samples_all)
            samples_all = samples_all[:sample_nums]
        # 数据增强，添加前后缀
        for sample in samples_all:
            self._add_prefix_suffix_in_sent(sample, positive)

        return samples_all

    def get_pos_samples(
        self,
        sample=False,
    ):
        """生成单功能点全量正样本."""
        samples = []
        for sent in self.sent_pats:
            samples.extend(self._get_sent_sample(sent, True))
        if sample:
            shuffle(samples)
            all_nums = len(samples)
            if not self.is_open_domain:
                sample_ratio = get_sample_ratio(
                    all_nums,
                    (1000, 5000, 10000, 30000, 50000),
                    (1, 0.8, 0.6, 0.5, 0.3, 0.2),
                )
            else:
                sample_ratio = get_sample_ratio(
                    all_nums,
                    (
                        100000,
                        500000,
                        1000000,
                        1500000,
                    ),
                    (
                        1,
                        0.8,
                        0.7,
                        0.6,
                        0.5,
                    ),
                )
            samples = samples[: math.floor(sample_ratio * all_nums)]
        return samples

    def get_neg_samples(
        self,
        sample=False,
    ):
        """生成单功能点全量负样本."""
        samples = []
        for sent in self.neg_sent_pats:
            samples.extend(self._get_sent_sample(sent, False))
        if sample:
            shuffle(samples)
            all_nums = len(samples)
            if not self.is_open_domain:
                sample_ratio = get_sample_ratio(
                    all_nums,
                    (1000, 5000, 10000, 30000, 50000),
                    (1, 0.8, 0.6, 0.5, 0.3, 0.2),
                )
            else:
                sample_ratio = 1
            samples = samples[: math.floor(sample_ratio * all_nums)]
        return samples


if __name__ == "__main__":
    args = parse_args()
    # 由raw_samples_2_std_json.py 生成的json文件.
    input_path = args.file_path
    # vehicle 或者 open_domain
    is_open_domain = args.is_open_domain
    open_domain_dict_path = "./files/open_domain_dict"
    open_domain_dict = (
        load_norm_dict(os.path.join(CURRENT_PATH, open_domain_dict_path))
        if is_open_domain
        else {}
    )
    # 正负样本存储路径
    pos_path = args.pos_path
    neg_path = args.neg_path
    # 日志文件
    log_file = args.log_path

    input_path = os.path.join(CURRENT_PATH, input_path)
    pos_path = os.path.join(CURRENT_PATH, pos_path)
    neg_path = os.path.join(CURRENT_PATH, neg_path)
    logging.basicConfig(
        level=logging.DEBUG,
        filename=os.path.join(CURRENT_PATH, log_file),
        filemode="w",
        format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",  # noqa: E501
    )
    with open(input_path, "r", encoding="utf-8") as f:
        data_all_lines = f.readlines()
    data_all = [json.loads(data.strip()) for data in data_all_lines]
    with open(pos_path, "w", encoding="utf-8") as f1, open(
        neg_path, "w", encoding="utf-8"
    ) as f2:
        for i in range(len(data_all)):
            func = data_all[i]["三级功能点"]
            bs = BaseSampler(data_all[i], is_open_domain)
            samples = bs.get_pos_samples(True)
            neg_samples = bs.get_neg_samples(True)
            logging.debug(
                "三级功能点:{}  正样本数量: {} ".format(
                    func.replace("\n", ""), len(samples)
                )
            )
            logging.debug(
                "三级功能点:{}  负样本数量: {} ".format(
                    func.replace("\n", ""), len(neg_samples)
                )
            )

            for sample in samples:
                f1.write(json.dumps(sample, ensure_ascii=False) + "\n")
            for sample in neg_samples:
                f2.write(json.dumps(sample, ensure_ascii=False) + "\n")
