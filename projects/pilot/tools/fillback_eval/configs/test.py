import time

from hdflow.eval import EVSEvalConfig
from project_utils import IssueConfig, PredictConfig

timestamp = time.strftime("%y%m%d%H%M", time.localtime())

name_prefix = "test_algo"

# 软件包, 软件包管理平台软件包名
app = "Pilot5.0-202304122330-V5.1_replace_hbm_202306081656"

# 软件包更新hbm文件, 支持：
# 1. bucket路径保存的hbm
# update_hbm = "dmpv2://matrix2/users/yilin.xiong/tmp/galaxy_v14_model.hbm"
# 2. aidi发版模型，通过 {name}:{version} 的形式来指定编译hbm，通常version为上一个版本
# 这里使用的即为aidi发版模型保存hbm,
# 对应 http://model.aidi.hobot.cc/models/publishedModel/43392?activeKey=3
update_hbm = "MCP5.0_galaxy_x3c_pe:v14.9.0"

# 软件包内部hbm更新相对路径
app_update_arc_path = "6V/adas/adas-rt/model_1920/model.hbm.frcnn"

# 集群资源(单个dataset消耗)
task_queue = "project-j5-pao-pilot5-idc-newage"
task_eval_queue = "svc-aip-cpu"
ipd_number = "PDT20220001"
num_worker = 20  # j5数量

# 运行环境
fillback_exe_image = "docker.hobot.cc/dlp/fillback-runtime:sdkplus.stable.lib"
fillback_sdk_image = "docker.hobot.cc/dlp/fillback-sdk:pilot5.sdkplus.stable"

# 其他配置， galaxy无odo pack
app_config = {"odo_type": "none"}

# 同时执行的回灌评测dataset数
num_parallel_job = 1

# evs回灌配置
evs = PredictConfig(
    meta=dict(
        task_queue=task_queue,
        task_type="perception_6v",
        task_device="J5",
        ipd_number=ipd_number,
        priority=5,
        runtime_config=dict(
            fillback_exe_image=fillback_exe_image,
            fillback_sdk_image=fillback_sdk_image,
        ),
        app_config=app_config,
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
    meta=dict(
        # common
        # refer to: http://10.10.112.88:8069/build/issue/regress.html#sphx-glr-build-issue-regress-py  # noqa
        is_statistics=False,
        execute_type=1,
        project_id=10032,  # for galaxy
        # sync_data=0, 只会拉取artifact建立时的初始issue
        # 避免不同版本的运行issue存在差异
        sync_data=0,
        priority=5,
        task_type=9,
        fillback_task_type="perception_6v",
        device_type="J5",
        project="PILOT5",
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
    )
)


# TODO: 比对评测报告
# compare_version = "xxx"
