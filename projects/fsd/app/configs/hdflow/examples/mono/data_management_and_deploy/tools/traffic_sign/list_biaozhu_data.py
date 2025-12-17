import datetime
import logging
import multiprocessing
import re

import pandas as pd
from hdflow.plugins.mono.badcase_process.common.clients import (
    build_context_manager,
)
from hdflow.plugins.mono.task_operations.traffic_sign.modules.label_platform_info import (  # noqa
    TrafficSignAnnotaionInfo,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)

context_manager = build_context_manager()


def _get_data_time(data_info):

    time_map = {
        "白天": "Day",
        "day": "Day",
        "Day": "Day",
        "夜晚": "Night",
        "night": "Night",
        "Night": "Night",
    }
    time = re.search(
        "|".join(time_map), data_info["task_name"] + str(data_info["task_tag"])
    )
    if time is not None:
        time = time_map[time.group()]
    else:
        time = "All"

    return time


def _get_data_location(data_info):
    location_map = {
        "中国": "CN",
        "CN": "CN",
        "cn": "CN",
        "美国": "US",
        "北美": "US",
        "US": "US",
        "us": "US",
    }

    location = re.search(
        "|".join(location_map.keys()),
        (
            data_info["task_doc"]
            + data_info["task_name"]
            + str(data_info["task_tag"])
        ),
    )
    if location is not None:
        location = location_map[location.group()]

    return location


def _get_data_purpose(data_info):
    test_tags = re.findall(
        "(评测|test|val|validation)",
        str(data_info["task_tag"]) + str(data_info["task_name"]),
    )
    train_tags = re.findall(
        "(训练|train)", str(data_info["task_tag"]) + str(data_info["task_name"])
    )

    if len(test_tags) > 0 and len(train_tags) > 0:
        purpose = "训练+评测"
    elif len(test_tags) > 0:
        purpose = "评测"
    else:
        purpose = "训练"

    return purpose


def _get_data_sensor(data_info):
    sensor_list = [
        "10635",
        "220",
        "0220",
        "323",
        "0323",
        "10652",
        "820",
        "0820",
        "x8b",
        "390",
        "D3-RCM",
        "D3RCM",
    ]
    sensors = re.findall(
        "(%s)" % "|".join(sensor_list), str(data_info["task_tag"]), flags=re.I
    )
    if len(sensors) < 1:
        sensors = re.findall(
            "(%s)" % "|".join(sensor_list),
            data_info["task_name"] + data_info["project_name"],
            flags=re.I,
        )
    if len(sensors) < 1:
        return None
    sensor = (sensors[0].lstrip("0")).upper()
    return sensor


def _get_ramp_data(data_info):
    ramp_flags = re.findall(
        "(ramp|匝道)",
        str(data_info["task_tag"])
        + str(data_info["task_name"])
        + str(data_info["project_name"]),
        flags=re.I,
    )
    if len(ramp_flags) > 0:
        ramp = "yes"
    else:
        ramp = "no"
    return ramp


def _get_badcase_data(data_info):
    badcase_flags = re.findall(
        "(badcase)",
        str(data_info["task_tag"])
        + str(data_info["task_name"])
        + str(data_info["project_name"]),
        flags=re.I,
    )
    if len(badcase_flags) > 0:
        badcase = "yes"
    else:
        badcase = "no"
    return badcase


def gen_data_id_entry(data_id_version_info):
    def format_entry(
        version_info,
        max_valid_version_info,
        max_anno_refine_version_info,
        history_version,
        sensor,
        time,
        location,
        purpose,
        is_ramp,
        is_badcase,
    ):
        entry = {
            "数据ID": version_info["data_id"],
            "数据版本": version_info["version"],
            "最大有效版本": max_valid_version_info.get("version", -1),
            "手工修改版本": max_anno_refine_version_info.get("version", -1),
            "AIDI_DATASET": version_info["aidi_dataset_id"],
            "最大有效版本AIDI_DATASET": max_valid_version_info.get(
                "aidi_dataset_id", None
            ),
            "手工修改版本AIDI_DATASET": max_anno_refine_version_info.get(
                "aidi_dataset_id", None
            ),
            "数据量(图)": version_info["image_num"],
            "数据量(框)": version_info.get("box_num", -1),
            "历史版本": history_version,
            "传感器": sensor,
            "数据时间": time,
            "国家": location,
            "用途": purpose,
            "是否匝道": is_ramp,
            "是否badcase": is_badcase,
            "标签": version_info["task_tag"],
            "标注文档": version_info["task_doc"],
            "任务名称": version_info["task_name"],
            "项目名称": version_info["project_name"],
            "任务ID": version_info["task_id"],
            "项目ID": version_info["project_id"],
            "上传日期": version_info["task_create_time"],
            "标注属性": version_info["attrParsed"],
            "质量反馈": version_info["task_feedback_score"],
            "项目类型": version_info["task_type"],
            "项目状态": version_info["task_status"],
            "项目号": version_info["pdt_project_id"],
        }
        return entry

    def formt_compact_entry(
        det_version_info,
        cls_version_info,
        max_valid_version_info,
        max_anno_refine_version_info,
        history_det_version,
        history_cls_version,
        sensor,
        time,
        location,
        purpose,
        is_ramp,
        is_badcase,
    ):
        cls_version = cls_version_info.get("version", -1)
        entry = {
            "数据ID": det_version_info["data_id"],
            "检测数据版本": det_version_info["version"],
            "分类数据版本": cls_version,
            "最大有效版本": max_valid_version_info.get("version", -1),
            "手工修改版本": max_anno_refine_version_info.get("version", -1),
            "检测AIDI_DATASET": det_version_info["aidi_dataset_id"],
            "分类AIDI_DATASET": cls_version_info.get("aidi_dataset_id", None),
            "最大有效版本AIDI_DATASET": max_valid_version_info.get(
                "aidi_dataset_id", None
            ),
            "手工修改版本AIDI_DATASET": max_anno_refine_version_info.get(
                "aidi_dataset_id", None
            ),
            "数据量(图)": det_version_info["image_num"],
            "数据量(框)": det_version_info.get("box_num", -1),
            "是否标注分类": "no" if cls_version == -1 else "yes",
            "传感器": sensor,
            "数据时间": time,
            "国家": location,
            "用途": purpose,
            "是否匝道": is_ramp,
            "是否badcase": is_badcase,
            "标签": det_version_info["task_tag"],
            "标注文档": det_version_info["task_doc"],
            "检测历史版本": history_det_version,
            "分类历史版本": history_cls_version,
            "检测任务名称": det_version_info["task_name"],
            "检测项目名称": det_version_info["project_name"],
            "检测任务ID": det_version_info["task_id"],
            "检测项目ID": det_version_info["project_id"],
            "检测上传日期": det_version_info["task_create_time"],
            "检测标注属性": det_version_info["attrParsed"],
            "检测质量反馈": det_version_info["task_feedback_score"],
            "检测项目类型": det_version_info["task_type"],
            "检测项目状态": det_version_info["task_status"],
            "分类任务名称": cls_version_info.get("task_name", ""),
            "分类项目名称": cls_version_info.get("project_name", ""),
            "分类任务ID": cls_version_info.get("task_id", ""),
            "分类项目ID": cls_version_info.get("project_id", ""),
            "分类上传日期": cls_version_info.get("task_create_time", ""),
            "分类标注属性": cls_version_info.get("attrParsed", ""),
            "分类质量反馈": cls_version_info.get("task_feedback_score", ""),
            "分类项目类型": cls_version_info.get("task_type", ""),
            "分类项目状态": cls_version_info.get("task_status", ""),
            "项目号": det_version_info["pdt_project_id"],
        }
        return entry

    det_info_list = []
    cls_info_list = []
    compact_info_list = []
    for _, version_info_entry in data_id_version_info.items():
        cls_version = version_info_entry.get("max_valid_cls_version", -1)
        det_version = version_info_entry.get("max_valid_det_version", -1)
        max_valid_version = version_info_entry.get("max_valid_version", -1)
        max_anno_refine_version = version_info_entry.get(
            "max_anno_refine_version", -1
        )
        history_det_version = version_info_entry["history_det_version"]
        history_cls_version = version_info_entry["history_cls_version"]
        version_info = version_info_entry["version_info"]

        cls_version_info = version_info.get(cls_version, dict())
        det_version_info = version_info.get(det_version, dict())
        max_valid_version_info = version_info.get(max_valid_version, dict())
        max_anno_refine_version_info = version_info.get(
            max_anno_refine_version, dict()
        )

        if not det_version_info:
            continue

        sensor = _get_data_sensor(det_version_info)
        time = _get_data_time(det_version_info)
        location = _get_data_location(det_version_info)
        purpose = _get_data_purpose(det_version_info)
        is_ramp = _get_ramp_data(det_version_info)
        is_badcase = _get_badcase_data(det_version_info)

        if det_version > 0:
            det_entry = format_entry(
                det_version_info,
                max_valid_version_info,
                max_anno_refine_version_info,
                history_det_version,
                sensor,
                time,
                location,
                purpose,
                is_ramp,
                is_badcase,
            )
            det_info_df = pd.DataFrame([det_entry])
            det_info_list.append(det_info_df)
        if cls_version > 0:
            cls_entry = format_entry(
                cls_version_info,
                max_valid_version_info,
                max_anno_refine_version_info,
                history_cls_version,
                sensor,
                time,
                location,
                purpose,
                is_ramp,
                is_badcase,
            )
            cls_info_df = pd.DataFrame([cls_entry])
            cls_info_list.append(cls_info_df)
        compact_entry = formt_compact_entry(
            det_version_info,
            cls_version_info,
            max_valid_version_info,
            max_anno_refine_version_info,
            history_det_version,
            history_cls_version,
            sensor,
            time,
            location,
            purpose,
            is_ramp,
            is_badcase,
        )
        compact_info_df = pd.DataFrame([compact_entry])
        compact_info_list.append(compact_info_df)

    return det_info_list, cls_info_list, compact_info_list


def list_data_worker(
    anno_info_getter,
    task_id,
    query_aidi_dataset=True,
    query_only_max_valid_data=False,
    sync_aidi_dataset=True,
    wait_aidi_sync=False,
):
    data_id_version_info = (
        anno_info_getter.query_data_id_valid_version_info_by_task(
            task_id,
            query_aidi_dataset=query_aidi_dataset,
            query_only_max_valid_data=query_only_max_valid_data,
            sync_aidi_dataset=sync_aidi_dataset,
            wait_aidi_sync=wait_aidi_sync,
        )
    )

    det_info, cls_info, compact_info = gen_data_id_entry(data_id_version_info)

    return det_info, cls_info, compact_info


def list_all_data_ids(
    pro_firm_type,
    save_path="",
    query_aidi_dataset=True,
    query_only_max_valid_data=False,
    sync_aidi_dataset=True,
    wait_aidi_sync=False,
    num_workers=8,
):
    ts_anno_info_getter = TrafficSignAnnotaionInfo()

    all_task_infos, all_task_ids = ts_anno_info_getter.query_tasks(
        pro_firm_type=pro_firm_type
    )
    pbar = tqdm(total=len(all_task_ids), unit="tasks")

    p = multiprocessing.Pool(num_workers)
    results = []
    for _, task_id in enumerate(all_task_ids):
        results.append(
            p.apply_async(
                list_data_worker,
                args=(
                    ts_anno_info_getter,
                    task_id,
                    query_aidi_dataset,
                    query_only_max_valid_data,
                    sync_aidi_dataset,
                    wait_aidi_sync,
                ),
                callback=lambda x: pbar.update(1),
            )
        )

    p.close()
    p.join()
    pbar.close()

    det_info_list = []
    cls_info_list = []
    compact_info_list = []
    for res in results:
        ret = res.get()
        if ret is not None:
            det_info_list += ret[0]
            cls_info_list += ret[1]
            compact_info_list += ret[2]

    with pd.ExcelWriter(save_path) as writer:
        if compact_info_list:
            logger.info("write compact ...")
            compact_info_df = pd.concat(compact_info_list)
            compact_info_df = compact_info_df.drop_duplicates(subset=["数据ID"])
            compact_info_df.to_excel(writer, sheet_name="汇总", index=False)
        if det_info_list:
            logger.info("write det ... ")
            det_info_df = pd.concat(det_info_list)
            det_info_df = det_info_df.drop_duplicates(subset=["数据ID"])
            det_info_df.to_excel(writer, sheet_name="标志牌检测", index=False)
        if cls_info_list:
            logger.info("write cls ... ")
            cls_info_df = pd.concat(cls_info_list)
            cls_info_df = cls_info_df.drop_duplicates(subset=["数据ID"])
            cls_info_df.to_excel(writer, sheet_name="标志牌识别", index=False)


if __name__ == "__main__":
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./stat_{current_time}.xlsx"
    with context_manager:
        list_all_data_ids(
            pro_firm_type="标志牌检测",
            save_path=output_file,
            query_aidi_dataset=True,
            query_only_max_valid_data=True,
            sync_aidi_dataset=True,
            wait_aidi_sync=False,
            num_workers=16,
        )
