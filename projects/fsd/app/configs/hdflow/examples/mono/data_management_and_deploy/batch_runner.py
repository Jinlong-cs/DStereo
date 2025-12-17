import json
import os
import subprocess
from dataclasses import dataclass
from getpass import getuser
from typing import List, Optional

import hdflow
from dataclasses_json import DataClassJsonMixin
from easydict import EasyDict
from hdflow.auto_dp.enum import AnnotateTags, FeedbackScore, TaskStatus

# -------------------------------------------------------
# structured inputs, dummy test, a yaml recommended
# -------------------------------------------------------
input_dicts = [
    # ---------------------------------------------------
    # packing examples
    # ---------------------------------------------------
    # 4pe detection
    dict(
        workflow="workflow_detection.py",
        task_name="vehicle_4pe",
        annoset_name="dummy_test",
        tags=["mono", "x8b", "train"],
        label_task_ids=None,
        label_dataset_ids=[
            "734214_2",
            "740722_2",
        ],
        purpose="train",
    ),
    dict(
        workflow="workflow_detection.py",
        task_name="ihbc_4pe",
        annoset_name="dummy_test",
        tags=["mono", "10652", "train"],
        label_task_ids=[37443],
        label_dataset_ids=None,
        purpose="train",
    ),
    dict(
        workflow="workflow_detection.py",
        task_name="vehicle_rear_4pe",
        annoset_name="dummy_test",
        tags=["mono", "10652", "train"],
        label_task_ids=[4765],
        label_dataset_ids=None,
        purpose="train",
    ),
    dict(
        workflow="workflow_detection.py",
        task_name="vehicle_rear_4pe",
        annoset_name="dummy_test_query_cond",
        tags=["mono", "10652", "train"],
        label_task_ids=None,
        label_dataset_ids=None,
        label_task_query_cond=[
            dict(
                type="ADD",
                pro_firm_number="PDT2020005",
                finish_date_start="2022-04-01",
                finish_date_end="2022-04-05",
                pro_firm_type="车尾检测",
                algorithm_user="姜妮",
                task_status=TaskStatus.DONE.value,
                feedback_score=FeedbackScore.ALL.value,
                tags=[AnnotateTags.PURPOSE_TRAIN.value],
            ),
            dict(
                type="REMOVE",
                pro_firm_number="PDT2020005",
                finish_date_start="2022-04-02",
                finish_date_end="2022-04-05",
                pro_firm_type="车尾检测",
                algorithm_user="姜妮",
                task_status=TaskStatus.DONE.value,
                feedback_score=FeedbackScore.ALL.value,
                tags=[AnnotateTags.PURPOSE_TRAIN.value],
            ),
        ],
        purpose="train",
    ),
    # person examples with roi list
    dict(
        workflow="workflow_detection.py",
        task_name="person_det_multitask",
        annoset_name="test_multitask_20220401-20220420",
        tags=["训练数据"],
        label_task_ids=None,
        label_dataset_ids=None,
        label_task_query_cond=[
            dict(
                type="ADD",
                pro_firm_number="PDT2020005",
                finish_date_start="2022-04-01",
                finish_date_end="2022-04-02",
                pro_firm_type="行人检测",
                algorithm_user="kaijun01.zhang",
                task_status=TaskStatus.DONE.value,
                feedback_score=FeedbackScore.ALL.value,
                tags=[AnnotateTags.PURPOSE_TRAIN.value],
            ),
        ],
        purpose="train",
    ),
    # 2pe detection
    dict(
        workflow="workflow_detection.py",
        task_name="person_age_classification",
        annoset_name="dummy_test_person_age",
        tags=["mono", "train"],
        label_task_ids=None,
        label_dataset_ids=["531343_4", "551144_2"],
        purpose="train",
    ),
    dict(
        workflow="workflow_detection.py",
        task_name="vehicle_2pe_detection",
        annoset_name="dummy_vehicle_2pe_detection_test",
        tags=["mono", "train", "x8b"],
        label_task_ids=["44901"],
        label_dataset_ids=None,
        purpose="train",
    ),
    # 4pe parsing
    dict(
        workflow="workflow_default_parsing.py",
        task_name="parsing_16cls",
        annoset_name="dummy_test_parsing_16cls",
        tags=["mono", "train"],
        label_task_ids=["39391"],
        label_dataset_ids=None,
        purpose="train",
    ),
    # 2pe cyclist keypoint
    dict(
        workflow="workflow_detection.py",
        task_name="cyclist_kps",
        annoset_name="dummy_test_kps",
        tags=["mono", "10652", "train"],
        label_task_ids=[40238],
        label_dataset_ids=None,
        purpose="train",
    ),
    dict(
        workflow="workflow_lane_parsing.py",
        task_name="lane_parsing_cn",
        annoset_name="dummy_test_lane_parsing_cn",
        tags=["mono", "train"],
        label_task_ids=["41010"],
        label_dataset_ids=None,
        purpose="train",
    ),
    dict(
        workflow="workflow_detection.py",
        task_name="vehicle_wheel_keypoints",
        annoset_name="dummy_test",
        tags=["mono", "X8B", "train"],
        label_task_ids=[46915],
        label_dataset_ids=None,
        purpose="train",
    ),
    # for road arrow test
    dict(
        workflow="workflow_detection.py",
        task_name="cn_road_arrow_4pe_CNX8B_roilist",
        annoset_name="dummy_test",
        tags=["mono", "CNX8B", "train", "roilist"],
        aidi_dataset_ids=[128939],
        day_or_night="night",
        purpose="train",
    ),
    dict(
        workflow="workflow_detection.py",
        task_name="cn_road_arrow_4pe_Badcase_roilist",
        annoset_name="dummy_test",
        tags=["mono", "CNX8B", "train", "roilist"],
        aidi_dataset_ids=[312709],
        day_or_night="day",
        purpose="train",
    ),
    dict(
        workflow="workflow_detection.py",
        task_name="cn_road_arrow_4pe",
        annoset_name="dummy_test",
        tags=["mono", "ar0820", "train"],
        label_task_ids=[39906],
        label_dataset_ids=None,
        purpose="train",
        day_or_night="day",
    ),
    dict(
        workflow="workflow_detection.py",
        task_name="cn_road_arrow_4pe_underpack",
        annoset_name="dummy_test",
        tags=["mono", "ar0820", "train"],
        label_task_ids=[39906],
        label_dataset_ids=None,
        purpose="train",
        day_or_night="day",
    ),
    dict(
        workflow="workflow_detection.py",
        task_name="cn_road_arrow_2pe_cls",
        annoset_name="dummy_test",
        tags=["mono", "ar0820", "train"],
        label_task_ids=[39906],
        label_dataset_ids=None,
        purpose="train",
        day_or_night="day",
    ),
    dict(
        workflow="workflow_detection.py",
        task_name="cn_road_arrow_2pe_fp",
        annoset_name="dummy_test",
        tags=["mono", "ar0820", "train"],
        label_task_ids=[39906],
        label_dataset_ids=None,
        purpose="train",
        day_or_night="day",
    ),
    dict(
        workflow="workflow_detection.py",
        task_name="cn_road_arrow_2pe_fp_prelabel",
        annoset_name="dummy_test",
        tags=["mono", "ar0820", "train"],
        label_task_ids=[39906],
        label_dataset_ids=None,
        purpose="train",
        day_or_night="day",
    ),
    dict(
        workflow="workflow_detection.py",
        task_name="cloud_model_traffic_sign_kps",
        annoset_name="traffic_sign_kps_cloudmodel",
        tags=["cloudmodel", "train"],
        label_task_ids=[5502],
        label_dataset_ids=None,
        config_root_dir="mono/data_management_and_deploy/densebox/configs_cloudmodel",  # noqa
        purpose="train",
    ),
    # ---------------------------------------------------
    # Evaluation Dataset examples
    # ---------------------------------------------------
    # for road arrow test
    dict(
        workflow="workflow_detection.py",
        task_name="cn_road_arrow_4pe_joint_narrow",
        annoset_name="dummy_test",
        tags=["sd", "x8b", "eval"],
        label_task_ids=[154751],
        purpose="eval",
    ),
    dict(
        workflow="workflow_detection.py",
        task_name="cn_road_arrow_4pe_joint",
        annoset_name="dummy_test",
        tags=["sd", "x8b", "eval"],
        label_task_ids=[143559],
        purpose="eval",
    ),
    dict(
        workflow="workflow_detection.py",
        task_name="cn_road_arrow_4pe",
        annoset_name="dummy_test",
        tags=["mono", "ar0820", "eval"],
        label_task_ids=[39906],
        label_dataset_ids=None,
        purpose="eval",
        day_or_night="day",
    ),
    dict(
        workflow="workflow_detection.py",
        task_name="cn_road_arrow_4pe_underpack",
        annoset_name="dummy_test",
        tags=["mono", "ar0820", "eval"],
        label_task_ids=[39906],
        label_dataset_ids=None,
        purpose="eval",
        day_or_night="day",
    ),
    dict(
        workflow="workflow_detection.py",
        task_name="cn_road_arrow_2pe_cls",
        annoset_name="dummy_test",
        tags=["mono", "ar0820", "eval"],
        label_task_ids=[39906],
        label_dataset_ids=None,
        purpose="eval",
        day_or_night="day",
    ),
    dict(
        workflow="workflow_detection.py",
        task_name="cn_road_arrow_2pe_fp",
        annoset_name="dummy_test",
        tags=["mono", "ar0820", "eval"],
        label_task_ids=[39906],
        label_dataset_ids=None,
        purpose="eval",
        day_or_night="day",
    ),
    dict(
        workflow="workflow_detection.py",
        task_name="cn_road_arrow_4pe_joint",
        annoset_name="dummy_test",
        tags=["mono", "ar0820", "eval"],
        label_task_ids=[39906],
        label_dataset_ids=None,
        purpose="eval",
        day_or_night="day",
    ),
    dict(
        workflow="workflow_detection.py",
        task_name="cn_road_arrow_4pe_joint_fp",
        annoset_name="dummy_test",
        tags=["mono", "ar0820", "eval"],
        label_task_ids=[39906],
        label_dataset_ids=None,
        purpose="eval",
        day_or_night="day",
    ),
    # 4pe parsing
    dict(
        workflow="workflow_lane_parsing.py",
        task_name="lane_parsing_cn",
        annoset_name="dummy_test_lane_parsing_cn",
        tags=["mono", "test"],
        label_task_ids=["55648"],
        label_dataset_ids=None,
        purpose="eval",
    ),
    dict(
        workflow="workflow_default_parsing.py",
        task_name="parsing_16cls",
        annoset_name="dummy_test_parsing_16cls",
        tags=["mono", "train"],
        source_root_dir=[
            "/jfs-public/smartauto/users/suyao.wang/"
            + "data/parsing-data-1080p/mono-0820-x8b/train/part6_badcase/data",
        ],
        label_task_ids=None,
        label_dataset_ids=None,
        purpose="eval",
    ),
    # 4pe detection
    dict(
        workflow="workflow_detection.py",
        task_name="vehicle_rear_4pe",
        annoset_name="dummy_test_rear",
        tags=["mono", "test"],
        label_task_ids=[4765],
        label_dataset_ids=None,
        purpose="eval",
    ),
    # classification
    dict(
        workflow="workflow_detection.py",
        task_name="person_age_classification",
        annoset_name="dummy_test_person_age",
        tags=["mono", "test"],
        label_task_ids=["31597"],
        label_dataset_ids=None,
        purpose="eval",
    ),
    # 2pe fp
    dict(
        workflow="workflow_detection.py",
        task_name="person_posneg_classification",
        annoset_name="dummy_test_person_posneg",
        tags=["mono", "test"],
        label_task_ids=["12179"],
        label_dataset_ids=None,
        purpose="eval",
    ),
    # # 2pe detection
    dict(
        workflow="workflow_detection.py",
        task_name="person_head_detection",
        annoset_name="dummy_test_person_head_detection",
        tags=["mono", "test"],
        label_task_ids=["16715"],
        label_dataset_ids=None,
        purpose="eval",
    ),
    # # 2pe vehicle wheel keypoints
    dict(
        workflow="workflow_detection.py",
        task_name="vehicle_wheel_keypoints",
        annoset_name="dummy_test_vehicle_wheel_keypoints",
        tags=["mono", "test"],
        label_task_ids=["46915"],
        label_dataset_ids=None,
        purpose="eval",
    ),
    # 2pe work condition scene
    dict(
        workflow="workflow_detection.py",
        task_name="scene_classification",
        annoset_name="dummy_test_wk_scene",
        tags=["mono", "train"],
        label_task_ids=None,
        label_dataset_ids=["591357_2", "592213_1"],
        purpose="train",
    ),
    # 2pe work condition weather
    dict(
        workflow="workflow_detection.py",
        task_name="weather_classification",
        annoset_name="dummy_test_wk_weather",
        tags=["mono", "train"],
        label_task_ids=None,
        label_dataset_ids=["591357_2", "592213_1"],
        purpose="train",
    ),
    # 2pe work condition lightstatus
    dict(
        workflow="workflow_detection.py",
        task_name="lightstatus_classification",
        annoset_name="dummy_test_wk_lightstatus",
        tags=["mono", "train"],
        label_task_ids=None,
        label_dataset_ids=["591357_2", "592213_1"],
        purpose="train",
    ),
    # 2pe work condition time
    dict(
        workflow="workflow_detection.py",
        task_name="time_classification",
        annoset_name="dummy_test_wk_time",
        tags=["mono", "train"],
        label_task_ids=None,
        label_dataset_ids=["591357_2", "592213_1"],
        purpose="train",
    ),
    # 2pe work IQA block
    dict(
        workflow="workflow_detection.py",
        task_name="block_classification",
        annoset_name="dummy_test_IQA_block",
        tags=["mono", "train"],
        label_task_ids=None,
        label_dataset_ids=["622213_3", "622214_2"],
        purpose="train",
    ),
    # 2pe work IQA blur
    dict(
        workflow="workflow_detection.py",
        task_name="blur_classification",
        annoset_name="dummy_test_IQA_blur",
        tags=["mono", "train"],
        label_task_ids=None,
        label_dataset_ids=["622213_3", "622214_2"],
        purpose="train",
    ),
    # 2pe work IQA glare
    dict(
        workflow="workflow_detection.py",
        task_name="glare_classification",
        annoset_name="dummy_test_IQA_glare",
        tags=["mono", "train"],
        label_task_ids=None,
        label_dataset_ids=["622213_3", "622214_2"],
        purpose="train",
    ),
    # 2pe vehicle_category_cls
    dict(
        workflow="workflow_detection.py",
        task_name="vehicle_category_2pe",
        annoset_name="vehicle_category_2pe_galaxy_front_eval_set",
        tags=["mono", "test"],
        label_task_ids=["83913", "83914", "83915", "83921"],
        label_dataset_ids=None,
        purpose="eval",
    ),
    # 2pe vehicle rear light classification
    dict(
        workflow="workflow_detection.py",
        task_name="vehicle_light_classification",
        annoset_name="train_vehicle_light_classification",
        tags=["mono", "train"],
        label_task_ids=None,
        label_dataset_ids=["594801_6"],
        purpose="train",
    ),
    # 2pe vehicle_left_door_cls
    dict(
        workflow="workflow_detection.py",
        task_name="vehicle_left_door_cls_2pe",
        annoset_name="vehicle_left_door_cls_2pe_galaxy_front_train_set",
        tags=["mono", "train"],
        label_task_ids=["128801"],
        label_dataset_ids=None,
        purpose="train",
    ),
    dict(
        workflow="workflow_detection.py",
        task_name="vehicle_left_door_cls_2pe",
        annoset_name="vehicle_left_door_cls_2pe_galaxy_front_eval_set",
        tags=["mono", "test"],
        label_task_ids=["128802"],
        label_dataset_ids=None,
        purpose="eval",
    ),
    # 2pe vehicle_right_door_cls
    dict(
        workflow="workflow_detection.py",
        task_name="vehicle_right_door_cls_2pe",
        annoset_name="vehicle_right_door_cls_2pe_galaxy_front_train_set",
        tags=["mono", "train"],
        label_task_ids=["128801"],
        label_dataset_ids=None,
        purpose="train",
    ),
    dict(
        workflow="workflow_detection.py",
        task_name="vehicle_right_door_cls_2pe",
        annoset_name="vehicle_right_door_cls_2pe_galaxy_front_eval_set",
        tags=["mono", "test"],
        label_task_ids=["128802"],
        label_dataset_ids=None,
        purpose="eval",
    ),
    # 2pe vehicle_trunk_door_cls
    dict(
        workflow="workflow_detection.py",
        task_name="vehicle_trunk_door_cls_2pe",
        annoset_name="vehicle_trunk_door_cls_2pe_galaxy_front_eval_set",
        tags=["mono", "test"],
        label_task_ids=["128802"],
        label_dataset_ids=None,
        purpose="eval",
    ),
    # 2pe vehicle occ
    dict(
        workflow="workflow_detection.py",
        task_name="vehicle_occlusion_2pe_no_prelabel",
        annoset_name="vehicle_occlusion_2pe_no_prelabel",
        tags=["test"],
        label_task_ids=[90572],
        label_dataset_ids=None,
        purpose="train",
    ),
    # 2pe vehicle fp
    dict(
        workflow="workflow_detection.py",
        task_name="2pe_vehicle_fp",
        annoset_name="2pe_vehicle_fp",
        tags=[
            "test",
        ],
        label_task_ids=[97125],
        label_dataset_ids=None,
        purpose="train",
    ),
    # 2pe vehicle rear occ
    dict(
        workflow="workflow_detection.py",
        task_name="vehicle_rear_occlusion_2pe_no_prelabel",
        annoset_name="vehicle_rear_occlusion_2pe_no_prelabel",
        tags=["test"],
        label_task_ids=[91576],
        label_dataset_ids=None,
        purpose="train",
        day_or_night="day",
    ),
    # 2pe vehicle rear fp
    dict(
        workflow="workflow_detection.py",
        task_name="2pe_vehicle_rear_fp",
        annoset_name="2pe_vehicle_rear_fp",
        tags=[
            "test",
        ],
        label_task_ids=[86964],
        label_dataset_ids=None,
        purpose="train",
    ),
]
# -------------------------------------------------------
# basics
# -------------------------------------------------------
submit_pre_envs = f"""
export PYTHONPATH=$(pwd)/deps/py_deps:$PYTHONPATH
export PYTHONPATH=$(pwd)/configs:$PYTHONPATH
export PYTHONPATH=$(pwd)/mono:$PYTHONPATH
export PATH=$(pwd)/deps/bin:$PATH
"""  # noqa

CPU_DOCKER = hdflow.get_docker_image("cpu")
GPU_DOCKER = hdflow.get_docker_image("cu102")


class AidiCfgs:
    # cluster in ["idc-v2", "aliyun-v2", "ucloud-v2", "bcloud"]
    idc_team_prelabel_small = EasyDict(
        cpu_queue_name="svc-aip-cpu",
        gpu_queue_name="team-prelabel-small",
        project_id="PDT20220004",
        cluster="idc-v2",
        engine="argo",
    )


@dataclass
class PackingSource(DataClassJsonMixin):
    task_name: str
    annoset_name: str
    tags: List[str]
    workflow: str
    # ["train", "eval"]
    purpose: str
    desc: Optional[str] = ""
    config_root_dir: Optional[str] = None
    aidi_dataset_ids: Optional[List[int]] = None
    label_task_ids: Optional[List[int]] = None
    label_dataset_ids: Optional[List[str]] = None
    label_task_query_cond: Optional[List[dict]] = None
    source_root_dir: Optional[List[str]] = None
    cam_locs: Optional[List[int]] = None
    # one of ["day", "night", "all"]
    day_or_night: Optional[str] = "all"

    def _check(self):
        if self.workflow in ["workflow_default_parsing.py"]:
            assert (
                self.aidi_dataset_ids
                == self.label_task_ids
                == self.label_dataset_ids
                == self.label_task_query_cond
                is None
            ), "default_parsing only support source_root_dir as source data."

    def make_default(self):
        self._check()
        # currently
        self.desc = (
            self.desc
            + f"|| Mono Packing, with tags:{'|'.join(self.tags)}\\n"
            + f"purpose: {self.purpose}\\n"
            + f"time: {self.day_or_night}\\n"
        )
        if self.aidi_dataset_ids:
            self.desc += f"aidi_dataset_ids: {'|'.join([str(i) for i in self.aidi_dataset_ids])}\\n"  # noqa
        if self.label_task_ids:
            self.desc += f"label_task_ids: {'|'.join([str(i) for i in self.label_task_ids])}\\n"  # noqa
        if self.label_dataset_ids:
            self.desc += f"label_dataset_ids: {'|'.join([str(i) for i in self.label_dataset_ids])}\\n"  # noqa
        if self.label_task_query_cond:
            self.desc += f"label_task_query_cond: {json.dumps(self.label_task_query_cond)}\\n"  # noqa
        if self.cam_locs:
            self.desc += (
                f"cam_locs: {'|'.join([str(i) for i in self.cam_locs])}\\n"
            )
        else:
            self.desc += "cam_locs: All\\n"
        self.desc = f"'{self.desc}'"


# -------------------------------------------------------------------------

instances = list()
for i in input_dicts:
    PackingSource.from_dict(i)
instances = [PackingSource.from_dict(i) for i in input_dicts]
retry_time = 1


def submit(
    submit_cache_dir,
    workflow_file_dir,
    aidicfg,
    local=False,
    test_mode=False,
):
    for instance in instances:
        instance.make_default()
        workflow_file = os.path.join(workflow_file_dir, instance.workflow)
        pipeline_name = (
            f"MonoDataDeployment_{instance.annoset_name}-{instance.purpose}"
        )
        executor = (
            "--local-executor simple" if local else "--remote-executor aidi"
        )

        submit_main = f"""
python3 execute.py \
{executor} \
--config {workflow_file} \
--cluster {aidicfg.cluster} \
--project-id {aidicfg.project_id} \
--engine {aidicfg.engine} \
--pipeline-name {pipeline_name} \
--submit-cache-dir {submit_cache_dir} \
--workflow-annoset-name {instance.annoset_name} \
--workflow-task-name {instance.task_name} \
--workflow-num-workers 8 \
--workflow-day-or-night {instance.day_or_night} \
--workflow-pass-invalid \
--workflow-trans-anno-to-pbrec \
--workflow-description {instance.desc} \
--workflow-tags {' '.join(map(str, instance.tags))} \
--workflow-purpose {instance.purpose} \
--workflow-cpu-docker {CPU_DOCKER} \
--workflow-gpu-docker {GPU_DOCKER} \
--workflow-cpu-queue-name {aidicfg.cpu_queue_name} \
--workflow-gpu-queue-name {aidicfg.gpu_queue_name} \
--viz-num-show 50 \
--workflow-viz-result \
"""

        cmd = submit_pre_envs + submit_main
        if instance.config_root_dir:
            cmd += (
                f"--workflow-config-root {instance.config_root_dir} "  # noqa
            )
        if instance.label_task_ids:
            cmd += f"--workflow-label-task-ids {' '.join(map(str, instance.label_task_ids))} "  # noqa
        if instance.aidi_dataset_ids:
            cmd += f"--workflow-aidi-dataset-ids {' '.join(map(str, instance.aidi_dataset_ids))} "  # noqa
        if instance.label_dataset_ids:
            cmd += f"--workflow-label-dataset-ids {' '.join(map(str, instance.label_dataset_ids))} "  # noqa
        if instance.source_root_dir:
            assert LOCAL_RUN is True, "Local path only support local run"
            cmd += f"--workflow-source-data-root-url {' '.join(map(str, instance.source_root_dir))} "  # noqa
        if instance.label_task_query_cond:
            jdict = json.dumps(instance.label_task_query_cond).replace(
                '"', '\\"'
            )
            cmd += f'--workflow-aidi-label-query-jdict "{jdict}" '  # noqa
        if instance.cam_locs:
            cmd += f"--workflow-target-camera-locs {' '.join(map(str, instance.cam_locs))} "  # noqa
        if test_mode:
            cmd += "--workflow-pipeline-test"
        print(cmd)
        subprocess.check_call(cmd, shell=True)


if __name__ == "__main__":
    TEST_MODE = os.getenv("TEST_MODE", "0") == "1"
    # LOCAL_RUN = False or TEST_MODE
    LOCAL_RUN = True

    submit(
        submit_cache_dir=f"/home/users/{getuser()}/data_hdfs/hdflow_cache",
        workflow_file_dir=f"mono/data_management_and_deploy/densebox",  # noqa
        aidicfg=AidiCfgs.idc_team_prelabel_small,
        local=LOCAL_RUN,
        test_mode=TEST_MODE,
    )
