import argparse
import copy
import datetime
import getpass
import json
import os
import uuid

import hdflow
import numpy as np
import yaml
from hatbc.aidi.dmp_client import DmpClient
from hatbc.aidi.pipeline.op import AIDIJobParam, aidi_job
from hatbc.auto_dp.database import (
    DEFAULT_CONFIG_PATH as DEFAULT_AUTO_DP_CONFIG_PATH,
)
from hatbc.auto_dp.database import DataBase
from hatbc.workflow import Constant, ControlFlow
from hatbc.workflow.trace import GraphTracer, get_traced_graph
from hdflow.auto_dp import get_annoset_dataiter, send_mail_about_annoset
from hdflow.auto_dp.enum import CameraLoc
from hdflow.data.anno_data_manage import AnnoDataManager
from hdflow.data.densebox.pipeline import (
    default_pack_and_upload,
    legacy_densebox_pack_and_upload,
    viz_densebox_packed_data_and_show,
)
from hdflow.misc import is_jenkins_test
from hdflow.plugins.mono.data_management_and_deploy.base.dm_client import (  # noqa
    build_context_manager,
)
from hdflow.plugins.mono.data_management_and_deploy.base.prelabel import (  # noqa
    packing_prelabel_pipeline,
)
from hdflow.plugins.mono.data_management_and_deploy.base.source_data_generator import (  # noqa
    packing_prelabel_annoset_iter,
    source_data_generator,
)
from hdflow.plugins.mono.data_management_and_deploy.base.structure import (  # noqa
    DataDeployTasksCfgs,
    LabelClientQueryTags,
    get_task_name2ts_configs,
    import_plugin_configs,
)
from hdflow.plugins.mono.data_management_and_deploy.eval_set.create_eval_set import (  # noqa
    delete_evalset,
    upload_evalset_and_setting,
)
from hdflow.plugins.mono.data_management_and_deploy.statistic import (  # noqa
    statistic_evalset,
    statistic_rec,
)
from hdflow.plugins.mono.models import (
    PrelabelTagging,
    load_models,
    load_predefines,
)
from hdflow.prediction.pipeline import InferLib
from hdflow.utils import dict_values_to_variable
from hdflow.utils.yaml import load_yaml_with_include_and_base

from auto_matrix.data.anno_transformer.anno import (  # isort:skip
    get_default_classinfo,
)

database_config = yaml.safe_load(open(DEFAULT_AUTO_DP_CONFIG_PATH, "r"))
context_resource_manager = build_context_manager()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workflow-annoset-name", type=str, default="example_vehicle_rear"
    )
    parser.add_argument("--workflow-task-name", type=str, default="NotSet")
    parser.add_argument("--workflow-cache-root", type=str, default="")
    parser.add_argument(
        "--workflow-label-task-ids", nargs="+", default=None, type=int
    )
    parser.add_argument(
        "--workflow-label-dataset-ids", nargs="+", default=None, type=str
    )
    parser.add_argument(
        "--workflow-aidi-dataset-ids", nargs="+", default=None, type=int
    )
    parser.add_argument(
        "--workflow-aidi-label-query-jdict", default=None, type=str
    )
    parser.add_argument(
        "--workflow-camera-loc-groups",
        nargs="+",
        type=str,
        default=CameraLoc.CAMERA_LOC_CamPinholeFront.name,
    )
    parser.add_argument(
        "--workflow-config-root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--workflow-purpose", type=str, choices=["train", "eval"]
    )
    parser.add_argument(
        "--workflow-prelabel-model",
        type=str,
        default="mono3.0_x8b_production_day",
    )
    parser.add_argument(
        "--workflow-day-or-night",
        type=str,
        default="all",
        choices=["day", "night", "all"],
    )
    parser.add_argument("--workflow-num-workers", type=int, default=8)
    parser.add_argument("--workflow-pass-invalid", action="store_true")
    parser.add_argument("--workflow-trans-anno-to-pbrec", action="store_true")
    parser.add_argument("--workflow-pipeline-test", action="store_true")
    parser.add_argument("--workflow-viz-result", action="store_true")
    parser.add_argument("--viz-num-show", type=int, default=10)
    parser.add_argument(
        "--workflow-description", type=str, default="train data"
    )
    parser.add_argument("--workflow-tags", nargs="+", default=[])
    parser.add_argument(
        "--workflow-cpu-docker",
        type=str,
        default=hdflow.get_docker_image("cpu"),
    )
    parser.add_argument(
        "--workflow-gpu-docker",
        type=str,
        default=hdflow.get_docker_image("cu102"),
    )
    parser.add_argument("--workflow-cpu-queue-name", type=str, default=None)
    parser.add_argument("--workflow-gpu-queue-name", type=str, default=None)
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3")
    parser.add_argument("--max-length", type=int, default=None)

    return parser.parse_known_args()[0]


args = parse_args()

dirname = os.path.dirname(os.path.relpath(__file__))

username = getpass.getuser()
timestr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
workflow_pipeline_name = (
    f"mono_packing_workflow_{timestr}_{uuid.uuid4().hex[:6]}"
)
config_root = (
    os.path.join(dirname, "configs")
    if args.workflow_config_root is None
    else args.workflow_config_root
)
TASK_NAME_TO_TS_CFG = get_task_name2ts_configs(config_root)
assert (
    args.workflow_task_name in TASK_NAME_TO_TS_CFG
), f"All possible:{TASK_NAME_TO_TS_CFG.keys()}, but given:{args.workflow_task_name}"  # noqa
task_configs: DataDeployTasksCfgs = TASK_NAME_TO_TS_CFG[
    args.workflow_task_name
]
if task_configs.task_ts_cfg is not None:
    with open(task_configs.task_ts_cfg) as fin:
        anno_ts_fn_config = yaml.load(fin, Loader=yaml.FullLoader)
else:
    anno_ts_fn_config = None
num_workers = args.workflow_num_workers

dmp_client: DmpClient = context_resource_manager.get(DmpClient)
anno_data_manager: AnnoDataManager = AnnoDataManager(
    dmp_client=context_resource_manager.get(DmpClient),
    db=context_resource_manager.get(DataBase),
)

cache_root = (
    dmp_client.dmp_url_to_local_path(args.workflow_cache_root)
    if dmp_client.is_dmp_url(args.workflow_cache_root)
    else args.workflow_cache_root
)
output_dir = os.path.join(
    cache_root, f"output/{workflow_pipeline_name}/packing_result"
)

if task_configs.packing_task_mode == "densebox":
    output_rec_path = f"{output_dir}/data.rec"
    output_anno_path = f"{output_dir}/data.json"
elif task_configs.packing_task_mode == "default":
    output_rec_path = f"{output_dir}/data.pb_rec"
    output_anno_path = f"{output_dir}/anno.pb_rec"
else:
    raise ValueError(
        f"[Packing][Detection]Packing mode do not support: "
        f"{task_configs.packing_task_mode}"
    )

output_roilist_path = f"{output_dir}/data_roilist.json"
viz_result_path = f"dmpv2://auto_tmp_2/users/{username}/{workflow_pipeline_name}/pack_data_viz_results"  # noqa
dmp_root_target_dir = (
    f"dmpv2://auto_tmp_2/users/{username}/CICDTEST/cached_packs"
    if is_jenkins_test()
    else f"dmpv2://mono/data/{args.workflow_task_name}/"
)
eval_setting_config = task_configs.eval_setting_config
evalset_config = task_configs.evalset_config

max_length = (
    20
    if (args.workflow_pipeline_test or is_jenkins_test())
    else args.max_length
)
need_prelabel = task_configs.prelabel_for_dataset


def dict_w_update(src, update_dict):
    src.update(update_dict)
    return src


# ---------------------------------------------------------------
# default prelabel configs
# ---------------------------------------------------------------
if need_prelabel:

    def _model_infos():
        model_infos = load_models(
            os.path.join(dirname, "../../common_utils/model_zoo.py"),
            query_model_info=dict(
                model_tag=args.workflow_prelabel_model,
                model_scopes=["common"],
            ),
        )
        model_predefines = load_predefines(
            os.path.join(dirname, "../../common_utils/model_zoo.py"),
        )
        assert len(model_infos) == 1
        model_info = model_infos[0]
        # w*h
        selected_image_shapes = list()
        for i in model_info["sensor"]:
            if i in model_predefines["SENSOR_NAME_TO_SHAPE"]:
                selected_image_shapes.append(
                    model_predefines["SENSOR_NAME_TO_SHAPE"][i]
                )
        model_config = load_yaml_with_include_and_base(
            model_info["config_path"]
        )
        return selected_image_shapes, model_config, model_info

    def _prelabel_configs(model_config, model_info):
        default_prelabel_configs = dict(
            name="prelabel_for_packing",
            output_dmp_dir=f"dmpv2://auto_tmp_2/users/{username}/prelabel/{workflow_pipeline_name}/prelabel_for_packing",  # noqa
            # noqa
            gpu_ids=(0,)
            if is_jenkins_test() or args.workflow_pipeline_test
            else tuple(map(int, args.gpu_ids.strip(",").split(","))),
            visual=False,
            visual_detection_threshold=0.3,
            num_worker=0
            if args.workflow_pipeline_test
            else 1
            if is_jenkins_test()
            else len(args.gpu_ids.strip(",").split(",")) * 2,
            machines=1,
            backend="concurrent_mp_spawn",
            tags=["packing_prelabel"],
            infer_lib=InferLib.HPFLOW_INFER,
            config=model_config,
            job_param=dict(
                docker_image=args.workflow_gpu_docker,
                queue_name=args.workflow_gpu_queue_name,
                walltime=60 * 24 * 14,  # one day default
                readonly_buckets=["auto_image", "auto_image_2", "adas"],
                writeable_buckets=["auto_tmp_2", "mono_aidi"],
            ),
        )
        update_g_tags_configs = dict(
            max_sample_num=max_length,
            info_keys=[model_info["tasks"]],
            prelabel_model_name=[
                PrelabelTagging.model_name_from_config(model_config)
            ],
            update_field="g_tags",
            result_record=False,
            job_param=dict(
                docker_image=args.workflow_cpu_docker,
                queue_name=args.workflow_cpu_queue_name,
                cpu=2,
                mem_per_cpu=2,
                readonly_buckets=[
                    "auto_image",
                    "auto_image_2",
                    "adas",
                    "mono",
                ],
                writeable_buckets=["auto_tmp_2", "mono_aidi"],
            ),
        )
        return default_prelabel_configs, update_g_tags_configs

    selected_image_shapes, model_config, model_info = _model_infos()
    default_prelabel_configs, update_g_tags_configs = _prelabel_configs(
        model_config, model_info
    )
else:
    default_prelabel_configs = None
    update_g_tags_configs = None
    selected_image_shapes = None


def get_postprocess_config():
    config = dict(type="Compose", post_processor_cfgs=[])
    if task_configs.data_pack_post_processer_config is not None:
        config["post_processor_cfgs"].append(
            task_configs.data_pack_post_processer_config
        )
    if args.workflow_trans_anno_to_pbrec:
        config["post_processor_cfgs"].append(
            dict(
                type="DenseboxRec2Pbrec",
                output_dir=output_dir,
                src_dataset_cfg=dict(
                    type="LegacyDenseBoxImageRecordDataset",
                    read_only=True,
                    to_rgb=False,
                    with_seg_label=False,
                    as_nd=False,
                    rec_path=output_rec_path,
                    anno_path=output_anno_path,
                ),
                dmp_root_target_dir=dmp_root_target_dir,
                collect_md5=True,
                num_workers=4,
                with_seg_label=False,
            )
        )
    return config


data_pack_post_processer_config = get_postprocess_config()
# ---------------------------------------------------------------------------------
#  make input configs
# ---------------------------------------------------------------------------------
query_conds = (
    [
        LabelClientQueryTags.from_dict(i)
        for i in json.loads(args.workflow_aidi_label_query_jdict)
    ]
    if args.workflow_aidi_label_query_jdict
    else None
)

default_data_packer_config = dict(
    type="LegacyDenseboxDataPacker",
    root_dir="./",
    output_densebox_rec_path=output_rec_path,
    output_densebox_json_path=output_anno_path,
    num_workers=num_workers,
)
if task_configs.use_roilist:
    default_data_packer_config[
        "output_densebox_roilist_json_path"
    ] = output_roilist_path

default_annotation_ts_config = dict(
    type="DenseBoxDetAnnoTs",
    verbose=False,
    config=anno_ts_fn_config,
    root_dir="./",
)

default_viz_dataset_config = dict(
    type="LegacyDenseBoxImageRecordDataset",
    rec_path=output_rec_path,
    anno_path=output_anno_path,
    read_only=False,
    with_seg_label=False,
    to_rgb=True,
    seg_label_dtype=np.uint8,
    as_nd=False,
)

default_viz_visualizer_config = dict(
    type="DenseBoxAnnoVisualizer",
    save_path=viz_result_path,
    viz_class_id=list(range(1, len(task_configs.class_name) + 1)),
    class_name=task_configs.class_name,
    lt_point_id=0,
    rb_point_id=2,
)

local_inputs = dict(
    num_worker=num_workers,
    output_dir=output_dir,
    dmp_root_target_dir=dmp_root_target_dir,
    desc=args.workflow_description,
    group_tags=args.workflow_tags,
    group_kv_tags=dict(
        task="detection",
        category=args.workflow_task_name,
        purpose=args.workflow_purpose,
    ),
    max_length=max_length,
    annoset_name=args.workflow_annoset_name,
    shuffle=False,
    collect_md5=True,
    to_account=list(
        set([f"{username}@horizon.ai"] + task_configs.task_owners_email)
    ),  # noqa
    subject=f"打包数据版本发布:{args.workflow_annoset_name}",
    # pack config
    use_roilist=task_configs.use_roilist,
    class_info_list=None
    if anno_ts_fn_config is None
    else get_default_classinfo(anno_ts_fn_config, point_num=10)
    if task_configs.classinfo_update_fn is None
    else task_configs.classinfo_update_fn(anno_ts_fn_config),
    data_packer_config=default_data_packer_config
    if task_configs.data_pack_update_fn is None
    else task_configs.data_pack_update_fn(
        default_data_packer_config,
        **{k: v for k, v in vars().items() if not k.startswith("__")},
    ),
    anno_transformer_config=default_annotation_ts_config
    if task_configs.anno_transformer_update_fn is None
    else task_configs.anno_transformer_update_fn(
        default_annotation_ts_config,
        **{k: v for k, v in vars().items() if not k.startswith("__")},
    ),
    prelabel_config=default_prelabel_configs
    if task_configs.prelabel_config_update_fn is None
    else task_configs.prelabel_config_update_fn(
        default_prelabel_configs,
        **{k: v for k, v in vars().items() if not k.startswith("__")},
    ),
    selected_image_shapes=selected_image_shapes,
    update_g_tags_configs=update_g_tags_configs,
    data_pack_post_processer_config=data_pack_post_processer_config,
    # for visualize
    viz_result_path=viz_result_path,
    viz_num_show=args.viz_num_show,
    viz_dataset_config=default_viz_dataset_config
    if task_configs.viz_dataset_config_update_fn is None
    else task_configs.viz_dataset_config_update_fn(
        default_viz_dataset_config,
        **{k: v for k, v in vars().items() if not k.startswith("__")},
    ),
    # funcation for visualization
    viz_fn_config=default_viz_visualizer_config
    if task_configs.viz_visualizer_config_update_fn is None
    else task_configs.viz_visualizer_config_update_fn(
        default_viz_visualizer_config,
        **{k: v for k, v in vars().items() if not k.startswith("__")},
    ),
    # create evalset config
    eval_setting_config=eval_setting_config,
    evalset_config=evalset_config,
    label_task_ids=args.workflow_label_task_ids,
    label_dataset_ids=args.workflow_label_dataset_ids,
    aidi_dataset_ids=args.workflow_aidi_dataset_ids,
    label_client_query=query_conds,
    pass_invalid=args.workflow_pass_invalid,
    dp_image_meta_keys=task_configs.dp_image_meta_keys,
    dp_pack_meta_keys=task_configs.dp_pack_meta_keys,
    statistic_anno_path=output_anno_path,
    # misc
    keep_duplicated=task_configs.keep_duplicate,
    packing_task_mode=task_configs.packing_task_mode,
)
anno_data_manager = Constant(anno_data_manager)

remote_inputs = copy.copy(local_inputs)
traced_inputs = copy.copy(local_inputs)
traced_inputs = dict_values_to_variable(traced_inputs)


def densebox_packing_pipeline(annoset, tags, extra_imports, prelabel_res):
    wait_tasks = [extra_imports]
    if prelabel_res is not None:
        wait_tasks.append(prelabel_res)
    with ControlFlow(wait=wait_tasks):
        # pack
        pack_info = legacy_densebox_pack_and_upload(
            annoset_query_str=None,
            annoset_ids=annoset.id,
            anno_transformer_config=traced_inputs["anno_transformer_config"],
            class_info_list=traced_inputs["class_info_list"],
            data_packer_config=traced_inputs["data_packer_config"],
            output_dir=traced_inputs["output_dir"],
            num_worker=traced_inputs["num_worker"],
            dmp_root_target_dir=traced_inputs["dmp_root_target_dir"],
            annoset_name=traced_inputs["annoset_name"],
            tags=tags["group_tags"],
            kv_tags=tags["group_kv_tags"],
            member_tags=tags["member_tags"],
            member_kv_tags=tags["member_kv_tags"],
            desc=traced_inputs["desc"],
            max_length=traced_inputs["max_length"],
            shuffle=traced_inputs["shuffle"],
            use_roilist=traced_inputs["use_roilist"],
            collect_md5=traced_inputs["collect_md5"],
            data_pack_post_processer_config=(
                traced_inputs["data_pack_post_processer_config"]
            ),
            keep_duplicated=traced_inputs["keep_duplicated"],
        )
    return pack_info


def default_packing_pipeline(annoset, tags, extra_imports, prelabel_res):
    wait_tasks = [extra_imports]
    if prelabel_res is not None:
        wait_tasks.append(prelabel_res)
    # pack
    with ControlFlow(wait=wait_tasks):
        pack_info = default_pack_and_upload(
            annoset_query_str=None,
            annoset_ids=annoset.id,
            anno_transformer_config=traced_inputs["anno_transformer_config"],
            data_packer_config=traced_inputs["data_packer_config"],
            output_dir=traced_inputs["output_dir"],
            num_worker=traced_inputs["num_worker"],
            dmp_root_target_dir=traced_inputs["dmp_root_target_dir"],
            annoset_name=traced_inputs["annoset_name"],
            tags=tags["group_tags"],
            kv_tags=tags["group_kv_tags"],
            desc=traced_inputs["desc"],
            max_length=traced_inputs["max_length"],
            shuffle=traced_inputs["shuffle"],
            member_tags=tags["member_tags"],
            member_kv_tags=tags["member_kv_tags"],
            collect_md5=traced_inputs["collect_md5"],
        )
    return pack_info


def prelabel_pipeline(annoset, extra_imports):
    job_result = packing_prelabel_pipeline(
        annoset=annoset,
        update_g_tags_configs=traced_inputs["update_g_tags_configs"],
        selected_image_shapes=traced_inputs["selected_image_shapes"],
        prelabel_config=traced_inputs["prelabel_config"],
        max_length=traced_inputs["max_length"],
        camera_loc_groups=args.workflow_camera_loc_groups,
        debug=(args.workflow_pipeline_test or is_jenkins_test()),
    )

    if GraphTracer.is_active():
        job_result.after(extra_imports)

    return job_result


def build_graph(imperative):
    with context_resource_manager, GraphTracer(imperative=imperative):
        extra_imports = import_plugin_configs(config_root)
        workflow_tails = [extra_imports]
        dataset_ids, label_tags = source_data_generator(
            traced_inputs,
            anno_data_manager,
            pass_invalid=traced_inputs["pass_invalid"],
            label_task_ids=args.workflow_label_task_ids,
            label_dataset_ids=args.workflow_label_dataset_ids,
            aidi_dataset_ids=args.workflow_aidi_dataset_ids,
            label_client_query=query_conds,
        )

        # update and merge tags
        tags = anno_data_manager.merge_tags(
            group_tags=traced_inputs["group_tags"],
            group_kv_tags=traced_inputs["group_kv_tags"],
            member_tags=traced_inputs.get("member_tags", None),
            member_kv_tags=traced_inputs.get("member_kv_tags", None),
            label_tags=label_tags,
        )
        # create annoset
        annoset = anno_data_manager.create_annoset(
            dataset_ids=dataset_ids,
            annoset_name=traced_inputs["annoset_name"],
            group_tags=tags["group_tags"],
            group_kv_tags=tags["group_kv_tags"],
            member_tags=tags["member_tags"],
            member_kv_tags=tags["member_kv_tags"],
            desc=traced_inputs["desc"],
        )
        if args.workflow_purpose in ["train"]:
            # prelabel
            prelabel_res = None
            if need_prelabel:
                prelabel_res = prelabel_pipeline(annoset, extra_imports)

            # pack
            if task_configs.packing_task_mode == "densebox":
                packing_pipeline = densebox_packing_pipeline
            elif task_configs.packing_task_mode == "default":
                packing_pipeline = default_packing_pipeline
            else:
                raise ValueError(
                    f"Error packing task mode:{task_configs.packing_task_mode}"
                )
            pack_info = packing_pipeline(
                annoset, tags, extra_imports, prelabel_res
            )

            workflow_tails.append(pack_info)

            statistic_infos = statistic_rec(
                pack_info,
                dp_image_meta_keys=traced_inputs["dp_image_meta_keys"],
                dp_pack_meta_keys=traced_inputs["dp_pack_meta_keys"],
                anno_path=traced_inputs["statistic_anno_path"],
                keep_duplicated=traced_inputs["keep_duplicated"],
                temp_data_cache_dir=output_dir,
                packing_task_mode=traced_inputs["packing_task_mode"],
            )
            workflow_tails.append(statistic_infos)

            update_result = anno_data_manager.update_annoset(
                annoset_id=annoset.id,
                train_pack_url=pack_info.get("url", None),
            )
            # pipeline test skip send mail
            if args.workflow_viz_result:
                image_viewer_url = viz_densebox_packed_data_and_show(
                    pack_info=pack_info,
                    output_image_dir=traced_inputs["viz_result_path"],
                    viz_dataset_config=traced_inputs["viz_dataset_config"],
                    viz_fn_config=traced_inputs["viz_fn_config"],
                    viz_num_show=traced_inputs["viz_num_show"],
                )
                image_viewer_url.after(update_result)
                workflow_tails.append(image_viewer_url)
                print(image_viewer_url._idata)
            elif not is_jenkins_test():
                send_mail = send_mail_about_annoset(
                    annoset,
                    traced_inputs["to_account"],
                    traced_inputs["subject"],
                    viz_url=None,
                )
                send_mail.after(update_result)
                workflow_tails.append(send_mail)
        elif args.workflow_purpose in ["eval"]:
            annoset_dataiter = get_annoset_dataiter(
                annoset_ids=annoset.id, max_length=traced_inputs["max_length"]
            )
            eval_set_id = upload_evalset_and_setting(
                annoset_dataiter=annoset_dataiter,
                evalset_config=traced_inputs["evalset_config"],
                setting_config=traced_inputs["eval_setting_config"],
                cache_data_url=traced_inputs["output_dir"],
                camera_loc=args.workflow_camera_loc_groups,
            )
            if GraphTracer.is_active():
                eval_set_id.after(extra_imports)

            statistic_infos = statistic_evalset(
                eval_set_id,
                temp_data_cache_dir=output_dir,
                upload_url=traced_inputs["dmp_root_target_dir"],
                dp_image_meta_keys=traced_inputs["dp_image_meta_keys"],
                dp_pack_meta_keys=traced_inputs["dp_pack_meta_keys"],
            )
            workflow_tails.append(statistic_infos)

            upload_status = anno_data_manager.update_annoset(
                annoset_id=annoset.id,
                eval_set_id=eval_set_id,
            )

            if is_jenkins_test():
                delete_result = delete_evalset(dataset_id=eval_set_id)
                delete_result.after(upload_status)
                workflow_tails.append(delete_result)
            else:
                send_mail = send_mail_about_annoset(
                    annoset,
                    traced_inputs["to_account"],
                    traced_inputs["subject"],
                    viz_url=None,
                )
                send_mail.after(upload_status)
                workflow_tails.append(send_mail)
        else:
            raise ValueError(
                f"Invalid Purpose: {args.workflow_purpose}, "
                f"all valid perpuse: [train, eval]"
            )

        workflow_job = aidi_job(
            tuple(workflow_tails),
            job_param=AIDIJobParam(
                task_id=f"pack_densebox_pbrec_{uuid.uuid4().hex[:10]}",
                task_type="filter",
                docker_image=args.workflow_cpu_docker,
                queue_name=args.workflow_cpu_queue_name,
                cpu=8,
                mem_per_cpu=8,
                walltime=60 * 24 * 7,
                readonly_buckets=[
                    "auto_image",
                    "auto_image_2",
                    "auto_eval",
                    "mono",
                    "auto_image_6",
                    "auto_image_4",
                ],
                writeable_buckets=["auto_tmp_2", "mono_aidi"],
            ),
        )

        workflow = get_traced_graph(workflow_job)
        return workflow


if __name__ == "__main__":
    workflow = build_graph(True)
else:
    workflow = build_graph(False)
