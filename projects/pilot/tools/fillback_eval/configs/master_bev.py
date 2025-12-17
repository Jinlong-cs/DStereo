import os
import time

from hdflow.eval import EVSEvalConfig
from project_utils import IssueConfig, PredictConfig

from projects.pilot.configs.project_utils.enum import BEVModelPackName

timestamp = time.strftime("%y%m%d%H%M", time.localtime())

name_prefix = "algo"

# 软件包, 软件包管理平台软件包名
app = "app_percep_2.0.0_DUO_20231216110716.zip"  # noqa
# 软件包更新hbm文件, 支持：
# 1. bucket路径保存的hbm
# update_hbm = "dmpv2://matrix2/users/yilin.xiong/tmp/galaxy_v14_model.hbm"
# 2. aidi发版模型，通过 {name}:{version} 的形式来指定编译hbm，通常version为上一个版本
# 这里使用的即为aidi发版模型保存hbm,
# 对应 http://model.aidi.hobot.cc/models/publishedModel/43392?activeKey=3
# 3. bev模型发版的gallery模型包
# 例如：http://gallery.hobot.cc/download/auto/pilot_algo/pilot5_1_model_release/byd_ek/project/snapshot/linux/arm/general/basic/pilot_perception_v1.1_ek_20231121-223310/byd_ek-pilot_perception_v1.1_ek_20231121-223310.zip # noqa

update_hbm = None
hbm_types = [
    BEVModelPackName.bev_large_range,
    BEVModelPackName.bev_small_range,
]
# 软件包内部hbm更新相对路径
app_update_arc_path = [
    os.path.join("app/adas/adas-rt/model", hbm_type) for hbm_type in hbm_types
]

# 集群资源(单个dataset消耗)
task_queue = "project-j5-pilot50-badcase-idc-newage"
task_eval_queue = "share-idc-newage-cpu"
ipd_number = "PDT20220001"
num_worker = 8  # j5数量

# 运行环境
fillback_exe_image = "docker.hobot.cc/dlp/fillback-runtime:stable"
fillback_sdk_image = "docker.hobot.cc/dlp/fillback-sdk:stable"

# 其他配置
app_config = {
    "fillback_type": "perception_11v_fusion_canraw_full",
    "disable_validator": 1,
    "bolepack": 1,
    "pack2csv": 1,
}

# 同时执行的回灌评测dataset数
num_parallel_job = 1

# evs回灌配置
evs = PredictConfig(
    meta=dict(
        task_queue=task_queue,
        task_type="common",
        task_device="J5",
        ipd_number=ipd_number,
        priority=5,
        runtime_config=dict(
            fillback_exe_image=fillback_exe_image,
            fillback_sdk_image=fillback_sdk_image,
        ),
        app_config=app_config,
        sdkplus=False,
    )
)

# evs评测配置
evs_eval = EVSEvalConfig(
    meta=dict(
        priority=5,
        task_queue=task_eval_queue,
        ipd_number=ipd_number,
    )
)

# issue回归评测配置
issue = IssueConfig(
    app=app,
    meta=dict(
        # common
        # refer to: http://10.10.112.88:8069/build/issue/regress.html#sphx-glr-build-issue-regress-py  # noqa
        is_statistics=False,
        execute_type=1,
        pdt_project_num="PDT20220001",
        # sync_data=0, 只会拉取artifact建立时的初始issue
        # 避免不同版本的运行issue存在差异
        sync_data=0,
        priority=5,
        task_type=9,
        fillback_task_type="common",
        device_type="J5",
        project="SD",
        ipd_number=ipd_number,
        task_queue=task_queue,
        common_params=app_config,
        spec_conf=dict(
            task_kwargs=dict(
                class_path="business.rule_regress.regress.RuleRegress"
            ),
            runtime_config=dict(
                fillback_exe_image=fillback_exe_image,
                fillback_sdk_image=fillback_sdk_image,
            ),
        ),
        sdkplus=False,
    ),
    fillback_device_num_per_worker=2,
)
