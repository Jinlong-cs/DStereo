import argparse
import copy
import datetime
import getpass
import json
import os
import socket
from uuid import uuid1

import yaml
from aidisdk.client import AIDIClient
from aidisdk.compute.package_abstract import GitCodeItem
from config.auto_threshold.default_thresholds import (
    default_pilot_model_threshold,
)
from config.project_info import (
    get_project_name,
    get_project_thresh_config,
    standardized_model_version,
)
from hatbc.adas_eval import EvaluationClient, ReportClient
from hatbc.aidi.git import GitCodePath
from hatbc.aidi.pipeline import ExperimentParams, WaitManualCheck
from hatbc.aidi.pipeline.asynctasks import AsyncWaitFinish
from hatbc.aidi.pipeline.op import AIDIJobParam, aidi_job
from hatbc.auto_dp.database import (
    DEFAULT_CONFIG_PATH as DEFAULT_AUTO_DP_CONFIG_PATH,
)
from hatbc.auto_dp.database import DataBase
from hatbc.filestream.bucket.client import BucketClient
from hatbc.filestream.file_helper import FileHelper
from hatbc.resource_manager import ContextResourceManager
from hatbc.workflow import get_traced_graph
from hatbc.workflow.trace import ControlFlow, GraphTracer
from hdflow.model_release.eval_report.model import PilotModelEvalReporter
from hdflow.model_release.eval_report.visualizer import (
    ModelReleaseReportViewer,
    ModelReportXLSViewer,
)
from hdflow.model_release.model_evaluation import PilotEvaluationReporter
from hdflow.model_release.model_threshold import (
    PilotModelReleasePreCheck,
    PilotModelThreshold,
)
from hdflow.utils.common import dict_values_to_variable, identity_pass, sub
from hdflow.utils.path import join

from projects.pilot.tools.production.nodes import (
    PilotHATPredictor,
    PilotHATTrainPipeline,
)


def parse_args():
    parser = argparse.ArgumentParser()
    # model info
    parser.add_argument(
        "--multi-model-type",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--multi-model-setting",
        required=True,
        type=str,
    )
    parser.add_argument("--multi-eval-data-setting", default=None, type=str)
    parser.add_argument(
        "--model-version",
        required=False,
        default=None,
        type=str,
    )
    parser.add_argument(
        "--multi-model-name-suffix",
        required=False,
        default=None,
        type=str,
    )
    parser.add_argument(
        "--multi-model-extra-info",
        required=False,
        default=None,
        type=str,
    )
    # env
    parser.add_argument("--git-branch", type=str, default=None)
    parser.add_argument("--git-commit-id", type=str, default=None)
    parser.add_argument("--git-tag", type=str, default=None)
    parser.add_argument(
        "--gpu-docker",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--cpu-docker",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--read-buckets",
        type=str,
        required=False,
        default="auto_eval",
    )
    parser.add_argument(
        "--write-buckets",
        type=str,
        required=False,
        default="pilot_tmp,matrix2",
    )
    # train only
    parser.add_argument(
        "--multi-train-config",
        required=False,
        default=None,
        type=str,
    )
    parser.add_argument(
        "--multi-train-stages",
        required=False,
        default=None,
        type=str,
    )
    parser.add_argument(
        "--multi-pretrain-name",
        required=False,
        default=None,
        type=str,
    )
    parser.add_argument(
        "--multi-pretrain-version",
        required=False,
        default=None,
        type=str,
    )
    parser.add_argument(
        "--multi-pretrain-stage",
        required=False,
        default=None,
        type=str,
    )
    parser.add_argument(
        "--multi-pretrain-checkpoint",
        required=False,
        default=None,
        type=str,
    )
    # eval only
    parser.add_argument(
        "--multi-eval-stage",
        required=False,
        default=None,
        type=str,
    )
    parser.add_argument(
        "--multi-eval-checkpoint",
        required=False,
        default=None,
        type=str,
    )
    parser.add_argument(
        "--multi-eval-name-suffix",
        required=False,
        default=None,
        type=str,
    )
    parser.add_argument(
        "--multi-diff-eval-name",
        required=False,
        default=None,
        type=str,
    )
    parser.add_argument(
        "--multi-diff-eval-name-release",
        required=False,
        default=None,
        type=str,
    )
    # cluster info
    parser.add_argument(
        "--queue-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--multi-train-queue",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--multi-train-resource",
        required=False,
        default=None,
        type=str,
        help="computing resource, format like '1x4' for 1 machine with 4 gpus",
    )
    parser.add_argument(
        "--multi-eval-resource",
        required=False,
        default=None,
        type=str,
        help="computing resource, format like '1x4' for 1 machine with 4 gpus",
    )
    parser.add_argument(
        "--multi-eval-queue",
        type=str,
        required=False,
    )
    parser.add_argument("--project-id", type=str, required=False, default=None)
    # feature status
    parser.add_argument("--disable-train", action="store_true", default=False)
    parser.add_argument("--enable-eval", action="store_true", default=False)
    parser.add_argument(
        "--multi-resume-mode",
        required=False,
        default=None,
        type=str,
    )
    parser.add_argument(
        "--enable-auto-threshold", action="store_true", default=False
    )
    parser.add_argument(
        "--enable-tracking", action="store_true", default=False
    )
    parser.add_argument("--enable-monitor", action="store_true", default=False)
    parser.add_argument("--pipeline-test", action="store_true", default=False)
    parser.add_argument(
        "--rerun-with-resume", action="store_true", default=False
    )
    parser.add_argument("--socket-port", type=int, default=8012)

    parser.add_argument(
        "--upload-metric-to-doris", action="store_true", default=False
    )
    parser.add_argument("--launcher", type=str, default="mpi")
    args, argv = parser.parse_known_args()
    if "--local-executor" in argv:
        return args, True
    else:
        return args, False


args, is_local = parse_args()

# model default meta
with open(f"{os.path.dirname(__file__)}/../../model_meta.yaml", "r") as r:
    model_meta = yaml.safe_load(r)

# cluster cfg
with open(f"{os.path.dirname(__file__)}/../pilot_cluster_cfg.yaml", "r") as r:
    cluster_cfg = yaml.safe_load(r)["base"]


def multi_model_separate(string: str, auto_pad: bool = False):
    if string == "null":
        return [None] * num_model
    if ";" in string:
        rets = string.split(";")
        rets = [None if ret == "null" else ret for ret in rets]
    elif auto_pad:
        assert (
            ";" not in string
        ), "Not support auto_pad=True for setting with ';'."
        rets = [string] * num_model
    else:
        rets = [string]
    assert len(rets) == num_model
    return rets


# using model_meta if args missing
# support specify models info by ';'
multi_model_type = args.multi_model_type.split(";")
num_model = len(multi_model_type)
multi_model_setting = multi_model_separate(
    args.multi_model_setting, auto_pad=True
)

if args.multi_train_stages is not None:
    multi_train_stages = multi_model_separate(args.multi_train_stages)
    assert (
        len(multi_train_stages) == num_model
    ), "multi_train_stages should be format like stage1,stage2;stage1,stage2 for multi model with ';' to separate"  # noqa
    _multi_train_stages = []
    for i, m in enumerate(multi_train_stages):
        if m is not None:
            _multi_train_stages.append(m.split(","))
        else:
            _multi_train_stages.append(
                model_meta[multi_model_type[i]]["train"]["stages"]
            )
    multi_train_stages = _multi_train_stages
else:
    multi_train_stages = [
        model_meta[m]["train"]["stages"] for m in multi_model_type
    ]

# using info in model_meta if not provide.
if args.multi_eval_stage is not None:
    multi_eval_stage = multi_model_separate(
        args.multi_eval_stage, auto_pad=True
    )
    _multi_eval_stage = []
    for i, m in enumerate(multi_eval_stage):
        if m is not None:
            _multi_eval_stage.append(m)
        else:
            _multi_eval_stage.append(
                model_meta[multi_model_type[i]]["train"]["stages"][-1]
            )
    multi_eval_stage = _multi_eval_stage
else:
    multi_eval_stage = [
        model_meta[m]["train"]["stages"][-1] for m in multi_model_type
    ]

# entry using info in model_meta.yaml
if args.multi_train_config is not None:
    multi_train_config = multi_model_separate(args.multi_train_config)
else:
    multi_train_config = [
        model_meta[m]["train"]["entry"] for m in multi_model_type
    ]

multi_eval_config = [model_meta[m]["eval"]["entry"] for m in multi_model_type]

if args.multi_train_resource is not None:
    _multi_train_resource = multi_model_separate(
        args.multi_train_resource, auto_pad=True
    )
    multi_train_resource = []
    for rs in _multi_train_resource:
        multi_train_resource.append(
            dict(
                num_machines=int(rs.split("x")[0]),
                num_gpus_per_machine=int(rs.split("x")[1]),
            )
        )
else:
    multi_train_resource = [
        model_meta[m]["train"]["resource"] for m in multi_model_type
    ]

if args.multi_eval_resource is not None:
    _multi_eval_resource = multi_model_separate(
        args.multi_eval_resource, auto_pad=True
    )
    multi_eval_resource = []
    for rs in _multi_eval_resource:
        multi_eval_resource.append(
            dict(
                num_machines=int(rs.split("x")[0]),
                num_gpus_per_machine=int(rs.split("x")[1]),
            )
        )
else:
    multi_eval_resource = [
        model_meta[m]["eval"]["resource"] for m in multi_model_type
    ]

for t, e in zip(multi_train_resource, multi_eval_resource):
    t["device_ids"] = list(range(t["num_gpus_per_machine"]))
    e["device_ids"] = list(range(e["num_gpus_per_machine"]))

multi_train_queue = [None] * num_model
multi_eval_queue = [None] * num_model
if not is_local:
    if not args.disable_train:
        multi_train_queue = multi_model_separate(
            args.multi_train_queue, auto_pad=True
        )
    if args.enable_eval:
        multi_eval_queue = multi_model_separate(
            args.multi_eval_queue, auto_pad=True
        )
else:
    # local dag, Only for placehold
    multi_train_queue = ["svc-aip-gpu"] * num_model
    multi_eval_queue = ["svc-aip-gpu"] * num_model

if args.gpu_docker is None:
    args.gpu_docker = cluster_cfg.get("docker_image")

if args.cpu_docker is None:
    args.cpu_docker = cluster_cfg.get("docker_image_cpu")

remote_inputs = {
    "multi_train_stages": multi_train_stages,
    "multi_train_config": multi_train_config,
    "multi_eval_config": multi_eval_config,
    "multi_train_resource": multi_train_resource,
    "multi_eval_resource": multi_eval_resource,
}

if args.multi_model_name_suffix is not None:
    multi_model_name_suffix = multi_model_separate(
        args.multi_model_name_suffix, auto_pad=True
    )
else:
    multi_model_name_suffix = [None] * num_model

if args.multi_eval_name_suffix is not None:
    multi_eval_name_suffix = multi_model_separate(
        args.multi_eval_name_suffix, auto_pad=True
    )
else:
    multi_eval_name_suffix = [None] * num_model

# support hybrid setting by info or ckpt cross models
# pretrain
multi_pretrain_checkpoint = [None] * num_model
if args.multi_pretrain_name is not None:
    multi_pretrain_name = multi_model_separate(args.multi_pretrain_name)
    multi_pretrain_version = multi_model_separate(args.multi_pretrain_version)
    multi_pretrain_stage = multi_model_separate(args.multi_pretrain_stage)
    for i, (n, v, s) in enumerate(
        zip(multi_pretrain_name, multi_pretrain_version, multi_pretrain_stage)
    ):
        if None in [n, v, s]:
            assert (
                n == v == s
            ), "pretrain_name, pretrain_version, pretrain_stage should be all null if one model dont need load pretrain ckpt!"  # noqa
        else:
            assert n is not None, "pretrain_name should provide!"
            assert v is not None, "pretrain_version should provide!"
            assert s is not None, "pretrain_stage should provide!"
            multi_pretrain_checkpoint[
                i
            ] = f"aidi_artifact://{n}/{s}/{v}/{s}-checkpoint-last.pth.tar"
if args.multi_pretrain_checkpoint is not None:
    multi_pretrain_url = multi_model_separate(args.multi_pretrain_checkpoint)
    for i in range(num_model):
        if multi_pretrain_url[i] is not None:
            assert (
                multi_pretrain_checkpoint[i] is None
            ), "pretrain info(name, version, stage) and checkpoint cannot be specified simultaneously!"  # noqa
            multi_pretrain_checkpoint[i] = multi_pretrain_url[i]

# resume mode
if args.multi_resume_mode is not None:
    multi_resume_mode = multi_model_separate(
        args.multi_resume_mode, auto_pad=True
    )
    multi_resume_mode = [
        True if m in ["true", "True"] else False for m in multi_resume_mode
    ]
else:
    multi_resume_mode = [False] * num_model

# eval
if args.multi_eval_checkpoint is not None:
    multi_eval_checkpoint = multi_model_separate(args.multi_eval_checkpoint)
else:
    multi_eval_checkpoint = [None] * num_model

# 适配版本火车
if args.multi_eval_data_setting:
    eval_data_settings = multi_model_separate(args.multi_eval_data_setting)
    for i, eval_data_setting in enumerate(eval_data_settings):
        if eval_data_setting is None:
            eval_data_settings[i] = [
                multi_model_setting[i].lower().replace("_lmdb", "")
            ]
        else:
            eval_data_settings[i] = [
                s.lower().replace("_lmdb", "")
                for s in eval_data_settings[i].split(",")
            ]
else:
    eval_data_settings = [
        [s.lower().replace("_lmdb", "")] for s in multi_model_setting
    ]

if args.enable_auto_threshold:
    if args.multi_diff_eval_name is not None:
        multi_diff_eval_name = multi_model_separate(args.multi_diff_eval_name)
    else:
        multi_diff_eval_name = [None] * num_model
    _multi_diff_eval_name = []
    for i, diff in enumerate(multi_diff_eval_name):
        if diff is None:
            _multi_diff_eval_name.append([None] * len(eval_data_settings[i]))
        else:
            _multi_diff_eval_name.append(diff.split(","))
    multi_diff_eval_name = _multi_diff_eval_name

    if args.multi_diff_eval_name_release is not None:
        multi_diff_eval_name_release = multi_model_separate(
            args.multi_diff_eval_name_release
        )
    else:
        multi_diff_eval_name_release = [None] * num_model
    _multi_diff_eval_name_release = []
    for i, diff in enumerate(multi_diff_eval_name_release):
        if diff is None:
            _multi_diff_eval_name_release.append(
                [None] * len(eval_data_settings[i])
            )
        else:
            _multi_diff_eval_name_release.append(diff.split(","))
    multi_diff_eval_name_release = _multi_diff_eval_name_release

model_version = args.model_version
model_version = standardized_model_version(model_version)

local_inputs = copy.deepcopy(remote_inputs)
# get traced inputs to construct DAG
traced_inputs = dict_values_to_variable(remote_inputs)

dag_project = get_project_name([[s.lower()] for s in multi_model_setting])

subproject_type = None
exp_label = []
if args.pipeline_test:
    exp_label.append("pipeline_test")
exp_info = dict(label=exp_label)
model_info = {}

dag_desc = dict(  # noqa
    task_scene="MODEL_RELEASE",
    domain="perception",
    model_version=model_version,
    subprojects=dag_project,
    exp_info=exp_info,
)
dag_desc = json.dumps(dag_desc)
assert (
    len(dag_desc) <= 255
), "len(dag_desc) should <= 255, please delete some dag info"

# code path
git_kwargs = dict(  # noqa
    repo="git@gitlab.hobot.cc:ptd/algorithm/ai-platform-algorithm/HAT.git",
    branch=args.git_branch,
    commit_id=args.git_commit_id,
    tag=args.git_tag,
)
hat_git_code = GitCodeItem(**git_kwargs)
code_path = GitCodePath(git_code=hat_git_code)

prefix = "pilot_multitask_"
prefix += "_".join(dag_project)
pipeline_name = prefix + "_" + model_version.replace(".", "_")
if args.pipeline_test:
    dmp_output_root = (
        f"dmpv2://pilot_tmp/users/{getpass.getuser()}/"
        f"model_diff_report/{pipeline_name}"
    )
else:
    dmp_output_root = (
        f"dmpv2://matrix2/users/{getpass.getuser()}/"
        f"model_diff_report/{pipeline_name}"
    )

timestr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
database_config = yaml.safe_load(open(DEFAULT_AUTO_DP_CONFIG_PATH, "r"))
context_resource_manager = ContextResourceManager(
    {
        DataBase: DataBase(config=database_config, tmp_dir="./tmp"),
        BucketClient: BucketClient(token=database_config["token"]),
        FileHelper: FileHelper(tmp_dir="./tmp"),
        AIDIClient: AIDIClient(token=database_config["token"]),
        EvaluationClient: EvaluationClient(),
        ReportClient: ReportClient(),
    }
)
dirname = os.path.relpath(os.path.dirname(__file__))
metric_config_path = os.path.join(dirname, "config/task_metrics.py")
release_metric_config_path = os.path.join(
    dirname, "config/task_metrics_release.py"
)

# aidi experiment manager
experiment_params = ExperimentParams(
    experiment_name=pipeline_name,
    experiment_path="dmpv2://matrix2/aidi_exp",
    tracking_enabled=True,
)

# set job level experiment params
job_exp_param_list = []
for i in range(num_model):
    job_exp_param_list.append(
        ExperimentParams(
            experiment_name="_".join(
                [prefix, multi_model_type[i], multi_model_setting[i]]
            ),
            run_name=model_version.replace(".", "_"),
            tracking_enabled=True,
        )
    )


# do init aidi experiment and run
if is_local:
    client = context_resource_manager.get(AIDIClient)
    if (
        experiment_params.tracking_enabled
        and not client.experiment.get_experiment(
            experiment_params.experiment_name
        )
    ):
        client.experiment.create_experiment(
            name=experiment_params.experiment_name,
            project_id=args.project_id,
            desc=experiment_params.desc,
            experiment_path=experiment_params.experiment_path,  # noqa
        )
    with client.experiment.init(
        experiment_params.experiment_name,
        "_".join([pipeline_name, uuid1().hex[:8]]),
        experiment_params.tracking_enabled,
    ) as run:
        run.log_runtime("local")

# using local ip.
if args.enable_monitor:
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    port = args.socket_port
    print(
        f"DAG will connect with hostname: {hostname}, ip: {ip}, port: {port}"
    )
    manual_check = WaitManualCheck(ip, port)

if args.rerun_with_resume:
    assert (
        args.enable_tracking
    ), "Please set --enable-tracking if rerun_with_resume=True!"

# get model train op and evaluate op (with auto_threshold)
train_op = []
job_descs = []
eval_op = []
check_op = [[] for _ in range(num_model)]
report_op = [[] for _ in range(num_model)]
diff_report_op = [[] for _ in range(num_model)]
upload_report_op = [[] for _ in range(num_model)]
visualize_op = [[] for _ in range(num_model)]
auto_threshold_op = [[] for _ in range(num_model)]
generate_report_op = [[] for _ in range(num_model)]
for i in range(num_model):
    job_desc = dict(
        model_type=multi_model_type[i],
        extra_name=None,
        extra_info=multi_model_setting[i],
    )
    job_descs.append(job_desc)
    if not args.disable_train:
        trainer = PilotHATTrainPipeline(
            model_version=model_version,
            model_name_postfix=multi_model_name_suffix[i],
            model_checkpoint=multi_pretrain_checkpoint[i],
            model_setting=multi_model_setting[i],
            enable_tracking=args.enable_tracking,
            tracking_version="v3",
            load_ckpt_resume_mode=multi_resume_mode[i],
            launcher=args.launcher,
        )
        train_op.append(trainer)
    if args.enable_eval:
        # evaluate (w/wo auto_threshold)
        # init evaluation operator
        evaluator = PilotHATPredictor(
            model_setting=multi_model_setting[i],
            model_version=model_version,
            model_name_postfix=multi_model_name_suffix[i],
            model_checkpoint=multi_eval_checkpoint[i],
            enable_tracking=args.enable_tracking,
            project_id=args.project_id,
            launcher=args.launcher,
        )
        eval_op.append(evaluator)
        if args.enable_auto_threshold:
            # for each eval setting
            for eval_setting in eval_data_settings[i]:
                threshold_config_file = get_project_thresh_config(
                    eval_setting, multi_model_type[i]
                )
                # init check
                check = PilotModelReleasePreCheck(
                    threshold_config_file=threshold_config_file,
                )
                check_op[i].append(check)
                # init reporter operator
                reporter = PilotEvaluationReporter(
                    to_account=getpass.getuser(),
                )
                report_op[i].append(reporter)
                # init model diff operator
                diff_reporter = PilotModelEvalReporter(num_processing=8)
                upload_reporter = PilotModelEvalReporter(num_processing=8)
                visualizer = ModelReportXLSViewer(
                    to_account=getpass.getuser(),
                    dmp_output_dir=os.path.join(dmp_output_root, "xls"),
                )
                diff_report_op[i].append(diff_reporter)
                upload_report_op[i].append(upload_reporter)
                visualize_op[i].append(visualizer)
                # init model thresh operator
                model_thresh = PilotModelThreshold(
                    task_config_file=threshold_config_file,
                    to_account=getpass.getuser(),
                    model_thresh=default_pilot_model_threshold.get(
                        (eval_setting, multi_model_type[i]), None
                    ),
                )
                auto_threshold_op[i].append(model_thresh)
                # init release report operator
                report_viewer = ModelReleaseReportViewer(
                    to_account=getpass.getuser(),
                    dmp_output_dir=os.path.join(
                        dmp_output_root, "release_report"
                    ),
                )
                generate_report_op[i].append(report_viewer)

# define workflow
with context_resource_manager, GraphTracer(imperative=False):
    # all jobs
    workflow_jobs = []
    training_metas = []
    for i in range(num_model):
        if is_local:
            # Just local working path when is_local=True
            working_path = os.getcwd()
        else:
            working_path = code_path()
        if not args.disable_train:
            training_meta = train_op[i](
                config_path=join(
                    working_path, traced_inputs["multi_train_config"][i]
                ),
                stages=traced_inputs["multi_train_stages"][i],
                device_ids=traced_inputs["multi_train_resource"][i][
                    "device_ids"
                ],
                num_machines=traced_inputs["multi_train_resource"][i][
                    "num_machines"
                ],
                pipeline_test=args.pipeline_test,
                working_path=working_path,
                working_env={"HAT_PILOT_MODEL_TYPE": multi_model_type[i]},
                rerun_with_resume=args.rerun_with_resume,
            )
            training_meta = aidi_job(
                training_meta,
                job_param=AIDIJobParam(
                    task_id=f"{multi_model_type[i]}_training_"
                    f"{multi_model_setting[i]}",
                    task_type="train",
                    docker_image=args.gpu_docker,
                    gpu=multi_train_resource[i]["num_gpus_per_machine"],
                    worker=multi_train_resource[i]["num_machines"],
                    walltime=60 if args.pipeline_test else 14 * 24 * 60,
                    readonly_buckets=args.read_buckets.split(","),
                    writeable_buckets=args.write_buckets.split(","),
                    queue_name=multi_train_queue[i],
                    desc=json.dumps(job_descs[i]),
                    git_codes=[hat_git_code],
                    experiment_params=job_exp_param_list[i],
                ),
            )
            workflow_jobs.append(training_meta)
        else:
            # build eval input by args
            training_meta = dict(
                newest_stage=dict(
                    stage=multi_eval_stage[i],
                )
            )
        training_metas.append(training_meta)

    if args.enable_monitor:
        with ControlFlow(wait=training_metas):
            check_res = manual_check()
        check_res = aidi_job(
            check_res,
            job_param=AIDIJobParam(
                task_id="manual_check_evaluation",
                task_type="filter",
                docker_image=args.cpu_docker,
                queue_name=args.queue_name,
                cpu=2,
                mem_per_cpu=4,
                walltime=60 if args.pipeline_test else 14 * 24 * 60,
                priority=cluster_cfg["priority"],
                git_codes=[hat_git_code],
                readonly_buckets=args.read_buckets.split(","),
                writeable_buckets=args.write_buckets.split(","),
            ),
        )

    for i in range(num_model):
        training_meta = training_metas[i]
        if args.enable_eval:
            training_newest_stage = training_meta["newest_stage"]["stage"]
            eval_name_suffix = "before"
            if multi_eval_name_suffix[i] is not None:
                eval_name_suffix = (
                    f"{multi_eval_name_suffix[i]}_" + eval_name_suffix
                )
            # evaluation
            for j, eval_data_setting in enumerate(eval_data_settings[i]):
                prefix = eval_data_setting
                if "crop" in multi_model_type[i]:
                    prefix = f"{eval_data_setting}_crop"
                dataset_filename = f"{prefix}_eval_datasets.py"
                dataset_config_file = (
                    f"projects/pilot/configs/datasets/{dataset_filename}"
                )
                if args.enable_auto_threshold:
                    # pre check
                    flag = check_op[i][j](
                        hat_code_path=working_path,
                        dataset_config_path=dataset_config_file,
                    )["hat_code_path"]

                if args.enable_monitor:
                    with ControlFlow(wait=[check_res, flag]):
                        evaluation = eval_op[i](
                            config_path=join(
                                working_path,
                                traced_inputs["multi_eval_config"][i],
                            ),
                            stage=training_newest_stage,
                            device_ids=traced_inputs["multi_eval_resource"][i][
                                "device_ids"
                            ],
                            num_machines=traced_inputs["multi_eval_resource"][
                                i
                            ]["num_machines"],
                            working_path=working_path,
                            prediction_name_suffix="_".join(
                                [eval_data_setting, eval_name_suffix]
                            ),
                            eval_data_setting=eval_data_setting,
                        )
                else:
                    evaluation = eval_op[i](
                        config_path=join(
                            working_path,
                            traced_inputs["multi_eval_config"][i],
                        ),
                        stage=training_newest_stage,
                        device_ids=traced_inputs["multi_eval_resource"][i][
                            "device_ids"
                        ],
                        num_machines=traced_inputs["multi_eval_resource"][i][
                            "num_machines"
                        ],
                        working_path=working_path,
                        prediction_name_suffix="_".join(
                            [eval_data_setting, eval_name_suffix]
                        ),
                        eval_data_setting=eval_data_setting,
                    )
                evaluation = aidi_job(
                    evaluation,
                    job_param=AIDIJobParam(
                        task_id=f"{multi_model_type[i]}_evaluation_"
                        f"{eval_data_setting}_before",
                        task_type="prediction",
                        docker_image=args.gpu_docker,
                        gpu=multi_eval_resource[i]["num_gpus_per_machine"],
                        worker=multi_eval_resource[i]["num_machines"],
                        walltime=60 if args.pipeline_test else 14 * 24 * 60,
                        priority=cluster_cfg["priority"],
                        readonly_buckets=args.read_buckets.split(","),
                        writeable_buckets=args.write_buckets.split(","),
                        queue_name=multi_eval_queue[i],
                        desc=json.dumps(job_descs[i]),
                        git_codes=[hat_git_code],
                        experiment_params=job_exp_param_list[i],
                    ),
                )
                workflow_jobs.append(evaluation)

                if args.enable_auto_threshold:
                    # report, before auto threshold
                    eval_dataset_config_path = join(
                        working_path, dataset_config_file
                    )

                    report, task = report_op[i][j](
                        prediction_name=evaluation["newest_stage"][
                            "prediction_name"
                        ],
                        diff_prediction_name=multi_diff_eval_name[i][j],
                        dataset_config_file=eval_dataset_config_path,
                        target_tasks=evaluation["newest_stage"]["task_names"],
                        project_id=args.project_id,
                        timestr=timestr,
                    )

                    async_wait = AsyncWaitFinish(async_run=True)
                    task = async_wait(task_async=task)
                    report, task = aidi_job(
                        report,
                        task,
                        job_param=AIDIJobParam(
                            task_id=f"{multi_model_type[i]}_create_report_"
                            f"{eval_data_setting}_before",  # noqa
                            task_type="filter",
                            docker_image=args.cpu_docker,
                            queue_name=args.queue_name,
                            cpu=2,
                            walltime=60
                            if args.pipeline_test
                            else 14 * 24 * 60,
                            priority=cluster_cfg["priority"],
                            git_codes=[hat_git_code],
                            readonly_buckets=args.read_buckets.split(","),
                            writeable_buckets=args.write_buckets.split(","),
                        ),
                    )

                    with ControlFlow(wait=task):
                        report = identity_pass(report)
                    report = aidi_job(
                        report,
                        job_param=AIDIJobParam(
                            task_id=f"{multi_model_type[i]}_get_report_"
                            f"{eval_data_setting}_before",  # noqa
                            task_type="filter",
                            docker_image=args.cpu_docker,
                            queue_name=args.queue_name,
                            cpu=2,
                            walltime=60
                            if args.pipeline_test
                            else 14 * 24 * 60,
                            priority=cluster_cfg["priority"],
                            git_codes=[hat_git_code],
                            readonly_buckets=args.read_buckets.split(","),
                            writeable_buckets=args.write_buckets.split(","),
                        ),
                    )

                    # diff report, before auto threshold
                    if args.upload_metric_to_doris:
                        model_scene = (
                            "day"
                            if "day" in eval_data_setting
                            else "night"
                            if "night" in eval_data_setting
                            else "parking"
                        )
                        ltc = get_project_name([[eval_data_setting]])[0]
                        upload_data = dict(
                            project_id=args.project_id,
                            ltc=ltc.upper(),
                            model_type=multi_model_type[i],
                            model_scene=model_scene,
                            model_setting=multi_model_setting[i],
                            model_version=args.model_version,
                        )
                        upload_report = upload_report_op[i][j](
                            dataset_config_file=eval_dataset_config_path,
                            metric_config_file=metric_config_path,
                            prediction_name=report["prediction_name"],
                            target_tasks=evaluation["newest_stage"][
                                "task_names"
                            ],
                            upload_data=upload_data,
                            is_upload_data=True if upload_data else False,
                        )
                        upload_result = aidi_job(
                            upload_report,
                            job_param=AIDIJobParam(
                                task_id=f"{multi_model_type[i]}_upload_report_"
                                f"{eval_data_setting}_before",  # noqa
                                task_type="filter",
                                docker_image=args.cpu_docker,
                                queue_name=args.queue_name,
                                cpu=2,
                                walltime=60
                                if args.pipeline_test
                                else 14 * 24 * 60,
                                priority=cluster_cfg["priority"],
                                git_codes=[hat_git_code],
                                readonly_buckets=args.read_buckets.split(","),
                                writeable_buckets=args.write_buckets.split(
                                    ","
                                ),
                            ),
                        )
                        workflow_jobs.append(upload_result)
                    else:
                        upload_data = None
                    cur_report = diff_report_op[i][j](
                        dataset_config_file=eval_dataset_config_path,
                        metric_config_file=metric_config_path,
                        prediction_name=report["prediction_name"],
                        target_tasks=evaluation["newest_stage"]["task_names"],
                    )
                    diff_report = diff_report_op[i][j](
                        dataset_config_file=eval_dataset_config_path,
                        metric_config_file=metric_config_path,
                        prediction_name=multi_diff_eval_name[i][j],
                        target_tasks=evaluation["newest_stage"]["task_names"],
                    )
                    diff_result = sub(cur_report, diff_report)
                    diff_result = visualize_op[i][j](
                        reports=[diff_result, cur_report, diff_report],
                        task_id_suffix=eval_data_setting,
                    )
                    diff_result = aidi_job(
                        diff_result,
                        job_param=AIDIJobParam(
                            task_id=f"{multi_model_type[i]}_model_eval_diff_"
                            f"{eval_data_setting}_before",  # noqa
                            task_type="filter",
                            docker_image=args.cpu_docker,
                            queue_name=args.queue_name,
                            cpu=2,
                            walltime=60
                            if args.pipeline_test
                            else 14 * 24 * 60,
                            priority=cluster_cfg["priority"],
                            git_codes=[hat_git_code],
                            readonly_buckets=args.read_buckets.split(","),
                            writeable_buckets=args.write_buckets.split(","),
                        ),
                    )

                    # auto get threshold
                    threshold = auto_threshold_op[i][j](
                        prediction_name=report["prediction_name"],
                        target_tasks=evaluation["newest_stage"]["task_names"],
                    )
                    threshold = aidi_job(
                        threshold,
                        job_param=AIDIJobParam(
                            task_id=f"{multi_model_type[i]}_model_threshold_"
                            f"{eval_data_setting}_before",
                            task_type="filter",
                            docker_image=args.cpu_docker,
                            queue_name=args.queue_name,
                            cpu=2,
                            walltime=60
                            if args.pipeline_test
                            else 14 * 24 * 60,
                            priority=cluster_cfg["priority"],
                            readonly_buckets=args.read_buckets.split(","),
                            writeable_buckets=args.write_buckets.split(","),
                            git_codes=[hat_git_code],
                        ),
                    )

                    # evaluation, after auto threshold
                    prediction_name_suffix = "release"
                    if multi_eval_name_suffix[i] is not None:
                        prediction_name_suffix = (
                            f"{multi_eval_name_suffix[i]}_"
                            + prediction_name_suffix
                        )

                    eval_dataset_config_path = join(
                        working_path, dataset_config_file
                    )

                    evaluation_release = eval_op[i](
                        config_path=join(
                            working_path,
                            traced_inputs["multi_eval_config"][i],
                        ),
                        stage=training_newest_stage,
                        model_thresh=threshold["model_threshold"],
                        device_ids=traced_inputs["multi_eval_resource"][i][
                            "device_ids"
                        ],
                        num_machines=traced_inputs["multi_eval_resource"][i][
                            "num_machines"
                        ],
                        working_path=working_path,
                        prediction_name_suffix="_".join(
                            [eval_data_setting, prediction_name_suffix]
                        ),
                        eval_data_setting=eval_data_setting,
                    )
                    evaluation_release = aidi_job(
                        evaluation_release,
                        job_param=AIDIJobParam(
                            task_id=f"{multi_model_type[i]}_evaluation_"
                            f"{eval_data_setting}_release",  # noqa
                            task_type="prediction",
                            docker_image=args.gpu_docker,
                            gpu=multi_eval_resource[i]["num_gpus_per_machine"],
                            worker=multi_eval_resource[i]["num_machines"],
                            walltime=60
                            if args.pipeline_test
                            else 14 * 24 * 60,
                            priority=cluster_cfg["priority"],
                            readonly_buckets=args.read_buckets.split(","),
                            writeable_buckets=args.write_buckets.split(","),
                            queue_name=multi_eval_queue[i],
                            desc=json.dumps(job_descs[i]),
                            git_codes=[hat_git_code],
                            experiment_params=job_exp_param_list[i],
                        ),
                    )

                    # report, after auto threshold
                    report_after, task_after = report_op[i][j](
                        prediction_name=evaluation_release["newest_stage"][
                            "prediction_name"
                        ],
                        diff_prediction_name=multi_diff_eval_name_release[i][
                            j
                        ],
                        dataset_config_file=eval_dataset_config_path,
                        target_tasks=evaluation_release["newest_stage"][
                            "task_names"
                        ],
                        project_id=args.project_id,
                        timestr=timestr,
                    )

                    async_wait = AsyncWaitFinish(async_run=True)
                    task_after = async_wait(task_async=task_after)
                    report_after, task_after = aidi_job(
                        report_after,
                        task_after,
                        job_param=AIDIJobParam(
                            task_id=f"{multi_model_type[i]}_create_report_"
                            f"{eval_data_setting}_release",  # noqa
                            task_type="filter",
                            docker_image=args.cpu_docker,
                            queue_name=args.queue_name,
                            cpu=2,
                            walltime=60
                            if args.pipeline_test
                            else 14 * 24 * 60,
                            priority=cluster_cfg["priority"],
                            git_codes=[hat_git_code],
                            readonly_buckets=args.read_buckets.split(","),
                            writeable_buckets=args.write_buckets.split(","),
                        ),
                    )

                    with ControlFlow(wait=task_after):
                        report_after = identity_pass(report_after)
                    report_after = aidi_job(
                        report_after,
                        job_param=AIDIJobParam(
                            task_id=f"{multi_model_type[i]}_get_report_"
                            f"{eval_data_setting}_release",  # noqa
                            task_type="filter",
                            docker_image=args.cpu_docker,
                            queue_name=args.queue_name,
                            cpu=2,
                            walltime=60
                            if args.pipeline_test
                            else 14 * 24 * 60,
                            priority=cluster_cfg["priority"],
                            git_codes=[hat_git_code],
                            readonly_buckets=args.read_buckets.split(","),
                            writeable_buckets=args.write_buckets.split(","),
                        ),
                    )

                    # diff report, after auto threshold
                    cur_report_after = diff_report_op[i][j](
                        dataset_config_file=eval_dataset_config_path,
                        metric_config_file=release_metric_config_path,
                        prediction_name=report_after["prediction_name"],
                        target_tasks=evaluation_release["newest_stage"][
                            "task_names"
                        ],
                    )
                    diff_report_after = diff_report_op[i][j](
                        dataset_config_file=eval_dataset_config_path,
                        metric_config_file=release_metric_config_path,
                        prediction_name=multi_diff_eval_name_release[i][j],
                        target_tasks=evaluation_release["newest_stage"][
                            "task_names"
                        ],
                    )
                    diff_result_after = sub(
                        cur_report_after, diff_report_after
                    )
                    diff_result_after = visualize_op[i][j](
                        reports=[
                            diff_result_after,
                            cur_report_after,
                            diff_report_after,
                        ],
                        task_id_suffix=eval_data_setting,
                    )
                    diff_result_after = aidi_job(
                        diff_result_after,
                        job_param=AIDIJobParam(
                            task_id=f"{multi_model_type[i]}_model_eval_diff_"
                            f"{eval_data_setting}_release",  # noqa
                            task_type="filter",
                            docker_image=args.cpu_docker,
                            queue_name=args.queue_name,
                            cpu=2,
                            walltime=60
                            if args.pipeline_test
                            else 14 * 24 * 60,
                            priority=cluster_cfg["priority"],
                            git_codes=[hat_git_code],
                            readonly_buckets=args.read_buckets.split(","),
                            writeable_buckets=args.write_buckets.split(","),
                        ),
                    )
                    if args.upload_metric_to_doris:
                        upload_report_after = upload_report_op[i][j](
                            dataset_config_file=eval_dataset_config_path,
                            metric_config_file=release_metric_config_path,
                            prediction_name=report_after["prediction_name"],
                            target_tasks=evaluation_release["newest_stage"][
                                "task_names"
                            ],
                            upload_data=upload_data,
                            is_upload_data=True if upload_data else False,
                        )

                        upload_result_after = aidi_job(
                            upload_report_after,
                            job_param=AIDIJobParam(
                                task_id=f"{multi_model_type[i]}_upload_report_"
                                f"{eval_data_setting}_release",  # noqa
                                task_type="filter",
                                docker_image=args.cpu_docker,
                                queue_name=args.queue_name,
                                cpu=2,
                                walltime=60
                                if args.pipeline_test
                                else 14 * 24 * 60,
                                priority=cluster_cfg["priority"],
                                git_codes=[hat_git_code],
                                readonly_buckets=args.read_buckets.split(","),
                                writeable_buckets=args.write_buckets.split(
                                    ","
                                ),
                            ),
                        )
                        workflow_jobs.append(upload_result_after)

                    # generate final report
                    release_report = generate_report_op[i][j](
                        report_name="_".join(
                            [
                                pipeline_name,
                                multi_model_type[i],
                                eval_data_setting,
                            ]
                        ),
                        report_before=report,
                        diff_report_before=diff_result,
                        model_threshold=threshold,
                        report_release=report_after,
                        diff_report_release=diff_result_after,
                        eval_data_settings=eval_data_settings[i][j],
                    )
                    release_report = aidi_job(
                        release_report,
                        job_param=AIDIJobParam(
                            task_id=f"{multi_model_type[i]}_model_release_report_"  # noqa
                            f"{eval_data_setting}",
                            task_type="filter",
                            docker_image=args.cpu_docker,
                            queue_name=args.queue_name,
                            cpu=2,
                            walltime=60
                            if args.pipeline_test
                            else 14 * 24 * 60,
                            priority=cluster_cfg["priority"],
                            readonly_buckets=args.read_buckets.split(","),
                            writeable_buckets=args.write_buckets.split(","),
                            git_codes=[hat_git_code],
                        ),
                    )
                    workflow_jobs.append(release_report)

    workflow = get_traced_graph(workflow_jobs)
