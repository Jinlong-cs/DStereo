import subprocess

import streamlit as st
from config.project_info import get_project_name
from hatbc.utils.socket import SocketServer

st.set_page_config(
    page_title="Pilot模型生产链路",
    page_icon=":underage:",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "# This is a header. This is an *extremely* cool app!",
    },
)

# sidebar
sidebar = st.sidebar
with sidebar:
    st.title("Pilot模型生产链路")
    st.markdown("---")
    st.header("workflow")
    step = st.multiselect(
        "模型生产包含节点",
        ["训练", "评测", "自动卡阈值评测"],
        max_selections=2,
        key="select_step",
        help="模型生成链路包含的节点，其顺序为训练->评测(可选自动卡阈值评测)。可通过该选项单独执行训练或评测。",
    )
    st.text_input(
        "workflow执行集群", value="svc-aip-cpu", key="queue_name", help="DAG执行集群"
    )
    st.text_input("版本号", placeholder="v0.0.1", key="version", help="发版版本号")
    st.text_input("project id", key="project_id", help="项目号，如PDT20220001")
    enable_tracking = st.checkbox(
        "开启实验管理",
        key="enable_tracking",
        help="是否开启实验管理功能，若开启，模型训练、评测会接入aidi实验管理",
    )
    pipeline_test = st.checkbox("pipeline test", key="pipeline_test")
    rerun_with_resume = st.checkbox(
        "模型训练自动断点恢复",
        key="rerun_with_resume",
        help="在开启实验管理的条件下，模型训练过程中的checkpoint会通过aidi实验管理以artifact的方式保存。"
        "若开启该选项，会自动搜索当前model_name是否存在artifact以及对应最新stage，并进一步地加载该stage"
        "和checkpoint，实现训练意外中断时的自动恢复训练功能。",
    )
    monitor = st.checkbox(
        "链路监控", key="pipeline_monitor", help="开启链路监控，能够对链路行为进行干预"
    )
    local_exec = st.checkbox("本地执行", key="local_exec", help="本地执行DAG")
    upload_metric_to_doris = st.checkbox(
        "上传评测指标到doris",
        key="upload_metric_to_doris",
        help="上传评测指标到doris,不要上传与发版无关的评测指标",
    )
    st.markdown("---")
    st.header("运行环境")
    git_type, git_seting = st.columns([0.7, 1])
    git_type.selectbox(
        "GIT type", ["branch", "commit_id", "tag"], index=0, key="git_type"
    )
    git_seting.text_input("setting", value="master", key="git_setting")
    st.text_input("GPU_DOCKER", key="gpu_docker", help="模型训练、评测节点docker")
    st.text_input(
        "CPU_DOCKER",
        key="cpu_docker",
        help="当使用自动卡阈值评测时，部分report节点需要该cpu_docker",
    )
    r_bucket, w_bucket = st.columns([1, 1])
    r_bucket.text_input("read buckets", value="auto_eval", key="read_buckets")
    w_bucket.text_input(
        "write buckets", value="pilot_tmp,matrix2", key="write_buckets"
    )


@st.cache_resource
def get_resource():
    num_model = 1
    return {"num_model": num_model}


resource = get_resource()

config, control = st.tabs(["模型配置", "链路监控"])

with control:
    port = st.text_input(
        "监控端口", placeholder=8012, help="监控端口，DAG将会与本地开发机(即DAG提交机器)端口进行通信，并实现监控"
    )
    check = st.radio("评测就绪", options=("YES", "NO"), index=1)

with config:
    container = st.container()
    add, delete = st.columns([9.5, 1])
    is_add = add.button("新增", help="增加一个需要生产的模型")
    if is_add:
        resource["num_model"] += 1
    block_key_format = "FormSubmitter:model_{}-确认"
    is_delete = delete.button("删除", help="删除上面一个模型")
    if is_delete:
        del st.session_state[
            block_key_format.format(resource["num_model"] - 1)
        ]
        resource["num_model"] -= 1

    n = resource["num_model"]

    # model list
    with container:
        for i in range(n):
            block = st.form(key=f"model_{i}")
            block.header(f"MODEL {i + 1}")
            model_setting, model_type = block.columns([1, 1])
            model_setting.text_input(
                "model setting",
                placeholder="c385_x3c_day",
                key=f"model_setting_{i}",
                help="模型setting，影响训练、评测节点使用的数据集",
            )
            model_type.text_input(
                "model type",
                placeholder="resize_2",
                key=f"model_type_{i}",
                help="模型type，对应hat/pilot/configs中模型类型，"
                "如resize_2、resize_4、crop等",
            )
            expd = block.expander("选填")
            expd.text_input(
                "model name suffix",
                key=f"model_name_suffix_{i}",
                help="模型名称后缀，影响训练保存的checkpoint名称",
            )
            if "训练" in step:
                block.markdown("---")
                block.markdown("#### 训练")
                (
                    train_queue,
                    train_num_machines,
                    train_num_gpus_per_machine,
                ) = block.columns([2, 1, 1])
                train_queue.text_input(
                    "集群",
                    key=f"train_queue_{i}",
                    help="训练节点使用的集群，需要GPU集群，如idc-share-titanxp-8",
                )
                train_num_machines.number_input(
                    "机器数",
                    value=1,
                    min_value=1,
                    max_value=8,
                    key=f"train_num_machine_{i}",
                )
                train_num_gpus_per_machine.number_input(
                    "机器GPU数",
                    value=1,
                    min_value=1,
                    max_value=8,
                    key=f"train_num_gpus_per_machine_{i}",
                )
                expd_train = block.expander("选填")
                expd_train.text_input(
                    "train stages",
                    key=f"train_stages_{i}",
                    help="模型训练stage list，通过','来分割，例如："
                    "with_bn,freeze_bn_1,freeze_bn_2",
                )
                (
                    checkpoint_type,
                    ckpt_setting,
                    resume_mode,
                ) = expd_train.columns([0.3, 0.8, 0.2])
                ckpt_type = checkpoint_type.selectbox(
                    "type",
                    ["url", "info"],
                    key=f"ckpt_type_{i}",
                    help="模型pretrain checkpoint提供方式，info通过"
                    "name、version、stage去到aidi实验管理load相应checkpoint artifact，"
                    "url通过一个能够访问的url链接直接提供模型地址。注意：更换type后，需要点击'确认'更新表格",
                )
                if ckpt_type == "url":
                    ckpt_setting.text_input(
                        "url",
                        placeholder="https://xxx",
                        key=f"train_checkpoint_{i}",
                    )
                else:
                    ckpt_name, ckpt_version, ckpt_stage = ckpt_setting.columns(
                        [1, 1, 1]
                    )
                    ckpt_name.text_input(
                        "pretrain name", key=f"pretrain_name_{i}"
                    )
                    ckpt_version.text_input(
                        "pretrain version", key=f"pretrain_version_{i}"
                    )
                    ckpt_stage.text_input(
                        "pretrain stage", key=f"pretrain_stage_{i}"
                    )
                resume_mode.checkbox("resume mode", key=f"resume_mode_{i}")
            if "评测" in step or "自动卡阈值评测" in step:
                block.markdown("---")
                block.markdown("#### 评测")
                (
                    eval_queue,
                    eval_num_machines,
                    eval_num_gpus_per_machine,
                ) = block.columns([2, 1, 1])
                eval_queue.text_input(
                    "集群",
                    key=f"eval_queue_{i}",
                    help="训练节点使用的集群，需要GPU集群，如idc-share-titanxp-8",
                )
                eval_num_machines.number_input(
                    "机器数",
                    value=1,
                    min_value=1,
                    max_value=8,
                    key=f"eval_num_machine_{i}",
                )
                eval_num_gpus_per_machine.number_input(
                    "机器GPU数",
                    value=1,
                    min_value=1,
                    max_value=8,
                    key=f"eval_num_gpus_per_machine_{i}",
                )
                if "自动卡阈值评测" in step:
                    diff_name, diff_name_release = block.columns([1, 1])
                    diff_name.text_input(
                        "diff eval name",
                        key=f"diff_eval_name_{i}",
                        help="当开启自动卡阈值评测后，需要提供对比的prediction name，"
                        "即aidi model评测平台leaderboard上对应预测名称。注意，若指定多"
                        "个eval data setting，该项通过','来分隔",
                    )
                    diff_name_release.text_input(
                        "diff eval name release",
                        key=f"diff_eval_name_release_{i}",
                        help="当开启自动卡阈值评测后，需要提供对比的prediction name release，"
                        "即aidi model评测平台leaderboard上对应预测名称。注意，若指定多"
                        "个eval data setting，该项通过','来分隔",
                    )
                expd_eval = block.expander("选填")
                eval_data_setting = expd_eval.text_input(
                    "eval data setting",
                    key=f"eval_data_setting_{i}",
                    help="评测setting，支持多个setting评测，请使用','分隔，"
                    "如galaxy_x02_side_day,galaxy_x02_side_night",
                )
                eval_stage, ckpt = expd_eval.columns([1, 2])
                eval_stage.text_input(
                    "eval stage",
                    key=f"eval_stage_{i}",
                    help="评测stage，若不提供，默认使用model_meta中的最后stage",
                )
                ckpt.text_input(
                    "checkpoint",
                    key=f"eval_checkpoint_{i}",
                    help="评测使用checkpoint，若不提供，默认使用name、version、"
                    "stage信息搜索artifact",
                )
                expd_eval.text_input(
                    "eval name suffix",
                    key=f"eval_name_suffix_{i}",
                    help="模型评测prediction名称，影响leaderboard上的名称",
                )
            block.form_submit_button("确认")

    st.markdown("---")
    Running = st.button(":runner: RUN", use_container_width=True)


def _check_settings():
    require_common_setting = [
        "queue_name",
        "project_id",
        "git_setting",
        "gpu_docker",
        "version",
    ]
    require_model_setting = [
        "model_type",
        "model_setting",
    ]
    if not local_exec:
        require_model_setting += [
            "train_queue",
            "eval_queue",
        ]
    for set in require_common_setting:
        assert len(st.session_state[set]) > 0, f"Please set {set}!"
    for set in require_model_setting:
        if set == "train_queue":
            if "训练" not in st.session_state["select_step"]:
                continue
        if set == "eval_queue":
            if (
                "评测" not in st.session_state["select_step"]
                and "自动卡阈值评测" not in st.session_state["select_step"]
            ):
                continue
        for i in range(n):
            assert (
                len(st.session_state[set + f"_{i}"]) > 0
            ), f"Please set {set} for MODEL {i}!"


def build_script():
    _check_settings()
    state = st.session_state
    args = {}
    action = []
    # common setting
    args["--config"] = "projects/pilot/tools/production/model_workflow.py"
    steps = state["select_step"]
    if "训练" not in steps:
        action.append("--disable-train")
    if "评测" in steps or "自动卡阈值评测" in steps:
        action.append("--enable-eval")
        if "自动卡阈值评测" in steps:
            action.append("--enable-auto-threshold")
    if enable_tracking:
        action.append("--enable-tracking")
    if rerun_with_resume:
        action.append("--rerun-with-resume")
    if pipeline_test:
        action.append("--pipeline-test")
    if upload_metric_to_doris:
        action.append("--upload-metric-to-doris")
    if monitor:
        action.append("--enable-monitor")
        assert len(port) > 0
        args["--socket-port"] = port
    action.append("--allow-wo-inputs")
    if local_exec:
        args["--local-executor"] = "simple"
    else:
        args["--remote-executor"] = "aidi"
    args["--queue-name"] = state["queue_name"]
    args["--project-id"] = state["project_id"]
    args[f"--git-{state['git_type'].replace('_', '-')}"] = state["git_setting"]
    args["--gpu-docker"] = state["gpu_docker"]
    if "cpu_docker" in state:
        args["--cpu-docker"] = state["cpu_docker"]
    args["--read-buckets"] = state["read_buckets"]
    args["--write-buckets"] = state["write_buckets"]

    def _get_multi_str(setting):
        _list = []
        for i in range(n):
            if f"{setting}_{i}" in state:
                value = str(state[f"{setting}_{i}"])
                if len(value) == 0:
                    _list.append("null")
                else:
                    _list.append(value)
            else:
                _list.append("null")
        return "\\;".join(_list)

    def _get_multi_bool(setting):
        _list = []
        for i in range(n):
            if f"{setting}_{i}" in state:
                value = state[f"{setting}_{i}"]
                if value:
                    _list.append("true")
                else:
                    _list.append("false")
            else:
                _list.append("false")
        return "\\;".join(_list)

    # model setting
    args["--model-version"] = state["version"]
    args["--multi-model-type"] = _get_multi_str("model_type")
    args["--multi-model-setting"] = _get_multi_str("model_setting")
    args["--multi-model-name-suffix"] = _get_multi_str("model_name_suffix")

    pipeline_name = "pilot_multitask_"
    setting = [
        item.split(",") if item != "null" else None
        for item in _get_multi_str("eval_data_setting").split("\\;")
    ]
    setting = [
        [state[f"model_setting_{idx}"]] if s is None else s
        for idx, s in enumerate(setting)
    ]
    dag_project = get_project_name(setting)
    pipeline_name += "_".join(dag_project)
    pipeline_name += "_" + state["version"].replace(".", "_")
    args["--pipeline-name"] = pipeline_name

    if "训练" in steps:
        args["--multi-train-stages"] = _get_multi_str("train_stages")
        args["--multi-train-queue"] = _get_multi_str("train_queue")
        multi_train_machine = _get_multi_str("train_num_machine")
        multi_train_gpus = _get_multi_str("train_num_gpus_per_machine")
        args["--multi-train-resource"] = "\\;".join(
            [
                m + "x" + g
                for m, g in zip(
                    multi_train_machine.split("\\;"),
                    multi_train_gpus.split("\\;"),
                )
            ]
        )
        args["--multi-pretrain-checkpoint"] = _get_multi_str(
            "train_checkpoint"
        )
        args["--multi-pretrain-name"] = _get_multi_str("pretrain_name")
        args["--multi-pretrain-version"] = _get_multi_str("pretrain_version")
        args["--multi-pretrain-stage"] = _get_multi_str("pretrain_stage")
        args["--multi-resume-mode"] = _get_multi_bool("resume_mode")

    if "评测" in steps or "自动卡阈值评测" in steps:
        args["--multi-eval-data-setting"] = _get_multi_str("eval_data_setting")
        args["--multi-eval-stage"] = _get_multi_str("eval_stage")
        args["--multi-eval-checkpoint"] = _get_multi_str("eval_checkpoint")
        args["--multi-eval-queue"] = _get_multi_str("eval_queue")
        multi_eval_machine = _get_multi_str("eval_num_machine")
        multi_eval_gpus = _get_multi_str("eval_num_gpus_per_machine")
        args["--multi-eval-resource"] = "\\;".join(
            [
                m + "x" + g
                for m, g in zip(
                    multi_eval_machine.split("\\;"),
                    multi_eval_gpus.split("\\;"),
                )
            ]
        )
        args["--multi-eval-name-suffix"] = _get_multi_str("eval_name_suffix")
        if "自动卡阈值评测" in steps:
            args["--multi-diff-eval-name"] = _get_multi_str("diff_eval_name")
            args["--multi-diff-eval-name-release"] = _get_multi_str(
                "diff_eval_name_release"
            )

    scripts = "python3 -c 'from hdflow.cli.execute import main, parse_args;args = parse_args();main(args)' "  # noqa  # noqa

    for k, v in args.items():
        scripts += f" {k} {v} \\\n"
    for v in action:
        scripts += f" {v} \\\n"

    return scripts


def main():
    script = build_script()
    st.info("Running shell script:")
    st.code(script, language="shellSession")
    try:
        with st.spinner("Wait for submit..."):
            subprocess.check_call(script, shell=True)
        st.info("Submit Done!")
    except BaseException as e:
        st.exception(e)
        st.info("Submit Fail!")


def connect_and_pass():
    assert len(port) > 0
    if f"socket_serve_{port}" in resource:
        serve = resource[f"socket_serve_{port}"]
    else:
        serve = SocketServer(port=int(port))
        resource[f"socket_serve_{port}"] = serve
    try:
        serve.wait_connect()
    except BaseException:
        pass
    serve.send_msg("pass")


if Running:
    main()

if check == "YES":
    connect_and_pass()
else:
    for k in resource.keys():
        if "socket_serve" in k:
            serve = resource[k]
            serve.close()
