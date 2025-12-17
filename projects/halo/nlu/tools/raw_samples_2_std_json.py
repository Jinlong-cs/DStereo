"""数据采集文档格式转换及格式检查脚本."""

__author__ = "ruosong.li@horizon.ai"
__date__ = "2022/06/06"
__copyright__ = "Copyright (C) 2022 Horizon Robotics, Inc."

import argparse
import itertools
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_PATH, "../../../../"))
from hat.core.nlu.nlu_utils import load_norm_dict  # noqa: E402


def parse_args():
    """输入参数解析."""
    parser = argparse.ArgumentParser(
        description="NLU transform raw samples to std json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f",
        "--file_path",
        type=str,
        default="./files/nlu语义_v1.0_训练数据采集_vehicle.csv",
        help="file path of data samping csv",
    )
    parser.add_argument(
        "-o",
        "--is_open_domain",
        type=bool,
        default=False,
        help="open domain flag",
    )
    args = parser.parse_args()
    return args


def is_nan(item):
    """Dataframe 元素判空."""
    return item != item


def df_preprocess(df):
    """对指定列存在空值的记录置为空字符串."""
    candi_columns = ["三级功能点", "领域", "意图", "槽位及可选值", "槽位值标注", "负样本句式"]
    for i in range(df.shape[0]):
        for column in candi_columns:
            if is_nan(df.loc[i, column]):
                df.loc[i, column] = ""


def single_sent_preprocess(sent_str):
    """单句句式标注预处理."""
    res = sent_str.strip()
    res = res.replace("｛", "{")
    res = res.replace("｝", "}")
    res = res.replace("{{", "{")
    res = res.replace("}}", "}")
    res = re.sub("[ \n\t]", "", res)
    return res


def sent_pat_process(sent_str):
    """句式标注预处理."""
    sents_list = sent_str.split("\n")
    sents_list = [
        single_sent_preprocess(item) for item in sents_list if item.strip()
    ]
    sents_list = list(set(sents_list))
    sents_expand_list = []
    # 并列槽位pattern检测： {xxx|xxx}
    pat = re.compile(r"\{[\w-]+\|[\w-]+(?:\|[\w-]+)*\}")
    for sent in sents_list:
        multi_slots_all = pat.findall(sent)
        if len(multi_slots_all) > 0:
            # 存在并列槽位,做槽位展开
            cur_replace_lists = []
            for cur_multi_slot in multi_slots_all:
                slots = cur_multi_slot[1:-1].split("|")
                cur_replace_lists.append(
                    [(cur_multi_slot, "{" + slot + "}") for slot in slots]
                )
            all_candi_replace = list(itertools.product(*cur_replace_lists))
            for cur_replace in all_candi_replace:
                cur_sent = sent
                for or_slot, replace_slot in cur_replace:
                    cur_sent = cur_sent.replace(or_slot, replace_slot)
                sents_expand_list.append(cur_sent)
        else:
            sents_expand_list.append(sent)
    return sents_expand_list


def slot_process(slot_str):
    """槽位语义定义处理."""
    slots_list = slot_str.split("\n")
    slots_list = [item.strip() for item in slots_list if item.strip()]
    res = defaultdict(list)
    for item in slots_list:
        assert "：" in item or ":" in item
        slot_name_str, slot_value_str = re.split("[：:]", item)
        slot_name = slot_name_str.strip()
        slot_value = re.split("[，,]", slot_value_str)
        slot_value = [item.strip() for item in slot_value if item.strip()]
        res[slot_name].extend(slot_value)
    return res


def slot_value_process(slot_v_str):
    """槽值标注处理."""
    slots_v_list = slot_v_str.split("\n")
    slots_v_list = [item.strip() for item in slots_v_list if item.strip()]
    res = {}
    for item in slots_v_list:
        k, v = re.split("[:：]", item)
        k = k.replace(" ", "")
        v_list = re.split("[、,， ]", v.strip())
        v_list = [i.strip().replace(" ", "") for i in v_list if i.strip()]
        res[k] = v_list
    return res


def sent_check(sent_str):
    """句式格式校验.

    1)如果存在未封闭的括号，则返回True  样例: value}  {value.
    2)存在重复的slot 请把{target}打开请把{target}打开.
    3)槽位中存在非法字符,即字母数字下划线和- 以外的字符
    """
    # 1)
    left_cnt = 0
    for i in range(len(sent_str)):
        cur = sent_str[i]
        if cur == "}":
            left_cnt -= 1
            if left_cnt < 0:
                return True
        if cur == "{":
            left_cnt += 1
    if left_cnt != 0:
        return True
    # 2)
    pat = re.compile(r"{[\w-]*}")
    slots_all = pat.findall(sent_str)
    if len(set(slots_all)) != len(slots_all):
        return True
    # 3)
    pat_all = re.compile(r"{.*?}")
    if len(slots_all) != len(pat_all.findall(sent_str)):
        return True
    else:
        return False


def sents_format_check(
    data_all, sents_key, is_open_domain=False, open_domain_dict_types=()
):
    """句式标注 批量格式校验."""
    errs = []
    pat = re.compile(r"{[\w-]*}")
    for data in data_all:
        func_name = data["三级功能点"]
        slot = data["slot"]
        sents_pat_all = data[sents_key]
        for sent in sents_pat_all:
            slots_all = pat.findall(sent)
            err_flag = sent_check(sent)
            for slot_name in slots_all:
                slot_type = (
                    slot_name[1:-1].split("-")[0]
                    if "-" in slot_name
                    else slot_name[1:-1]
                )
                if not is_open_domain:
                    if slot_type not in slot:
                        err_flag = True
                else:
                    if (slot_type not in slot) and (
                        slot_type not in open_domain_dict_types
                    ):
                        err_flag = True
            if err_flag:
                errs.append({"三级功能点": func_name, "问题句式": sent})
    return errs


def slots_val_format_check(
    data_all, is_open_domain=False, open_domain_dict_types=()
):
    """槽位值标注 批量格式校验."""
    sample_slot = {
        "value": ["百分数", "分数", "小数", "整数", "电台数值"],
        "time": ["时间数值"],
    }
    errs = []
    for data in data_all:
        func_name = data["三级功能点"]
        slot = data["slot"]
        slot_value = data["slot_value"]
        for k in slot_value:
            try:
                s, v = k.split("@")
                if s in slot and v in slot[s]:
                    continue
                else:
                    errs.append(
                        {"三级功能点": func_name, "问题槽位值": k, "错误类型": "与标准槽值定义不匹配"}
                    )
            except Exception:
                errs.append(
                    {
                        "三级功能点": func_name,
                        "问题槽位值": k,
                        "错误类型": "槽值格式错误，不满足“槽位@标准值”格式",
                    }
                )
        if not is_open_domain:
            for k in slot:
                std_values = slot[k]
                for std_value in std_values:
                    if k in sample_slot and std_value in sample_slot[k]:
                        continue
                    elif k + "@" + std_value not in slot_value:
                        errs.append(
                            {
                                "三级功能点": func_name,
                                "问题槽位值": k,
                                "错误类型": "槽值*{}*未提供标注值".format(std_value),
                            }
                        )
        else:
            slot_value_types = {k.split("@")[0] for k in slot_value}
            for k in slot:
                std_values = slot[k]
                if (
                    (k not in slot_value_types)
                    and (k not in open_domain_dict_types)
                    and not (
                        k in sample_slot
                        and all(
                            [
                                std_value in sample_slot[k]
                                for std_value in std_values
                            ]
                        )
                    )
                ):
                    errs.append(
                        {
                            "三级功能点": func_name,
                            "问题槽位值": k,
                            "错误类型": "槽位*{}*未提供标注值".format(k),
                        }
                    )

    return errs


def save_err_report(
    save_path,
    format_errs,
    sents_errs,
    slot_value_errs,
    neg_sents_errs,
):
    """生成并保存错误报告."""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("数据采集格式错误报告：\n")
        f.write("报告内主要错误类型说明：\n")
        f.write("1.数据加载错误(一般为槽值标注存在格式问题)\n")
        f.write(
            "2.正样本句式解析错误1)句式中{}内的变量在该功能点的语义定义中不存在 2)句式中括号未封闭 3)句式中存在重复槽位 \n"
        )
        f.write(
            "3.槽位值解析错误1)与标准槽值定义不匹配 2)槽值标注不满足“槽位@标准值”格式 3)槽值未提供采样函数或标注结果 \n"
        )
        f.write("4.负样本句式解析错误（同句式解析错误）\n\n")
        f.write("错误数量统计：\n")
        f.write("数据加载错误数量：" + str(len(format_errs)) + "\n")
        f.write("正样本句式解析错误数量：" + str(len(sents_errs)) + "\n")
        f.write("槽位值解析错误数量：" + str(len(slot_value_errs)) + "\n")
        f.write("负样本句式解析错误数量：" + str(len(neg_sents_errs)) + "\n\n")
        f.write("错误记录：\n")
        f.write("\n**数据加载错误：\n")
        for item in format_errs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.write("\n**正样本句式解析错误：\n")
        for item in sents_errs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.write("\n**槽位值解析错误：\n")
        for item in slot_value_errs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.write("\n**负样本句式解析错误：\n")
        for item in neg_sents_errs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    args = parse_args()
    # 数据采集文件，需转换成csv格式
    file_path = os.path.join(CURRENT_PATH, args.file_path)
    # vehicle 或者 open_domain
    is_open_domain = args.is_open_domain
    save_path = file_path + "_std_format.json"
    err_report_path = file_path + "_err_report.json"
    open_domain_dict_types = ()

    if is_open_domain:
        base_path = os.path.join(CURRENT_PATH, "./files")
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        download_cmd = "bash download_opendomain_dict.sh"
        ret = os.system(download_cmd)
        if ret != 0:
            raise UserWarning("Error occurs while execute:" + download_cmd)
        open_domain_dict_path = os.path.join(base_path, "./open_domain_dict")
        open_domain_dict = load_norm_dict(open_domain_dict_path)
        open_domain_dict_types = tuple(open_domain_dict.keys())

    data_all = []
    format_errs = []
    pos_sents_cnt = 0
    neg_sents_cnt = 0

    df = pd.read_csv(file_path, header=0)
    df_preprocess(df)
    lines_num = df.shape[0]
    for i in range(lines_num):
        try:
            cur_res = {}
            cur_res["三级功能点"] = df.loc[i, "三级功能点"]
            cur_res["domain"] = df.loc[i, "领域"].strip()
            cur_res["intent"] = df.loc[i, "意图"].strip()
            cur_res["slot"] = slot_process(df.loc[i, "槽位及可选值"])
            sent_pats = sent_pat_process(df.loc[i, "正样本句式"])
            neg_sent_pats = sent_pat_process(df.loc[i, "负样本句式"])
            cur_res["sents_pattern"] = sent_pats
            cur_res["neg_sents_pattern"] = neg_sent_pats
            pos_sents_cnt += len(sent_pats)
            neg_sents_cnt += len(neg_sent_pats)
            cur_res["slot_value"] = slot_value_process(df.loc[i, "槽位值标注"])
            data_all.append(cur_res)
        except Exception as e:
            format_errs.append({"三级功能点": df.loc[i, "三级功能点"], "err": str(e)})

    sent_errs = sents_format_check(
        data_all, "sents_pattern", is_open_domain, open_domain_dict_types
    )
    neg_sent_errs = sents_format_check(
        data_all, "neg_sents_pattern", is_open_domain, open_domain_dict_types
    )
    slot_val_errs = slots_val_format_check(
        data_all, is_open_domain, open_domain_dict_types
    )
    save_err_report(
        err_report_path,
        format_errs,
        sent_errs,
        slot_val_errs,
        neg_sent_errs,
    )
    print("正样本句式数：", str(pos_sents_cnt))
    print("负样本句式数：", str(neg_sents_cnt))

    with open(save_path, "w", encoding="utf-8") as f:
        for data in data_all:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
