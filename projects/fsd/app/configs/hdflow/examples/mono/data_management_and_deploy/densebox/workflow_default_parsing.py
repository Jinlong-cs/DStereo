import argparse
import copy
import datetime
import getpass
import os
from collections import defaultdict

import hdflow
import numpy as np
from hatbc.aidi.dmp_client import DmpClient
from hatbc.aidi.pipeline.op import AIDIJobParam, aidi_job
from hatbc.auto_dp.database import (  # noqa
    DEFAULT_CONFIG_PATH as DEFAULT_AUTO_DP_CONFIG_PATH,
)
from hatbc.auto_dp.database import DataBase
from hatbc.workflow import Constant, ControlFlow
from hatbc.workflow.trace import GraphTracer, get_traced_graph
from hdflow.auto_dp import send_mail_about_annoset
from hdflow.auto_dp.enum import CameraLoc
from hdflow.data.anno_data_manage import AnnoDataManager
from hdflow.data.densebox.pipeline import (
    get_parsing_merge_info,
    legacy_densebox_pack_and_upload,
    viz_densebox_packed_data_and_show,
)
from hdflow.misc import is_jenkins_test
from hdflow.pipeline.utils import append_locs
from hdflow.plugins.mono.data_management_and_deploy.base.dm_client import (  # noqa
    build_context_manager,
)
from hdflow.plugins.mono.data_management_and_deploy.base.source_data_generator import (  # noqa
    source_data_generator,
)
from hdflow.plugins.mono.data_management_and_deploy.base.structure import (  # noqa
    DataDeployTasksCfgs,
    LabelClientQueryTags,
    get_task_name2ts_configs,
    import_plugin_configs,
)
from hdflow.plugins.mono.data_management_and_deploy.eval_set.create_eval_set import (  # noqa
    upload_evalset_and_setting,
)
from hdflow.plugins.mono.data_management_and_deploy.statistic import (  # noqa
    statistic_evalset,
    statistic_rec,
)
from hdflow.plugins.mono.data_management_and_deploy.task_operations.parsing.get_prelabel_tasks import (  # noqa
    get_render_images_dataset_id_map,
)
from hdflow.plugins.mono.data_management_and_deploy.task_operations.parsing.misc import (  # noqa
    get_image_ann_pair_generator,
)
from hdflow.prediction.pipeline import (
    InferLib,
    parallel_prelabel_and_write_json,
)
from hdflow.utils import dict_values_to_variable
from hdflow.utils.io import get_simple_json_dataiter_form_dir
from hdflow.utils.path import to_unique_local_dir
from hdflow.utils.yaml import load_yaml_with_include_and_base

from auto_matrix.config import Config  # isort:skip
from auto_matrix.data.anno_transformer.seg.anno_ts import (  # isort:skip
    default_anno_to_contours_fn,
)
from auto_matrix.data.anno_transformer.seg.colormap import (  # isort:skip
    get_colors_and_class_names_for_parsing,
)
from auto_matrix.data.densebox.anno.anno import ClassInfo  # isort:skip

context_resource_manager = build_context_manager()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workflow-annoset-name",
        type=str,
        default="example_annoset_default_parsing",
    )
    parser.add_argument("--workflow-task-name", type=str, default="NotSet")
    parser.add_argument("--workflow-cache-root", type=str, default="")
    parser.add_argument(
        "--workflow-source-data-root-url", nargs="+", default=None, type=str
    )
    parser.add_argument(
        "--workflow-target-camera-locs", nargs="+", default=None, type=int
    )
    parser.add_argument(
        "--workflow-config-root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--workflow-purpose",
        type=str,
        choices=["train", "eval"],
        default="train",
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
    parser.add_argument("--workflow-as-gpu-job", action="store_true")
    parser.add_argument(
        "--workflow-camera-loc-groups",
        nargs="+",
        type=str,
        default=CameraLoc.CAMERA_LOC_CamPinholeFront.name,
    )
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

    return parser.parse_known_args()[0]


args = parse_args()
assert args.workflow_purpose in ["train", "eval"]
action = "Packing" if args.workflow_purpose in ["train"] else "CreatEvalSet"
dirname = os.path.dirname(os.path.relpath(__file__))
username = getpass.getuser()
timestr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
workflow_pipeline_name = f"mono_{action}_workflow_{timestr}"
config_root = (
    os.path.join(dirname, "configs")
    if args.workflow_config_root is None
    else args.workflow_config_root
)
TASK_NAME_TO_TS_CFG = get_task_name2ts_configs(config_root)
assert args.workflow_task_name in TASK_NAME_TO_TS_CFG
task_configs: DataDeployTasksCfgs = TASK_NAME_TO_TS_CFG[
    args.workflow_task_name
]
num_worker = args.workflow_num_workers
dmp_client: DmpClient = context_resource_manager.get(DmpClient)
anno_data_manager = AnnoDataManager(
    dmp_client=context_resource_manager.get(DmpClient),
    db=context_resource_manager.get(DataBase),
)
assert (
    task_configs.keep_duplicate is False
), "Parsing Do not support keep_duplicate"  # noqa
cache_root = (
    dmp_client.dmp_url_to_local_path(args.workflow_cache_root)
    if dmp_client.is_dmp_url(args.workflow_cache_root)
    else args.workflow_cache_root
)
output_dir = os.path.join(
    cache_root, f"output/{workflow_pipeline_name}/{action}_result"
)
output_rec_path = f"{output_dir}/data.rec"
output_anno_path = f"{output_dir}/data.json"
viz_result_path = f"dmpv2://auto_tmp_2/user/{username}/{workflow_pipeline_name}/pack_data_viz_results"  # noqa
dmp_root_target_dir = (
    f"dmpv2://auto_tmp_2/user/{username}/CICDTEST/cached_{action}"
    if is_jenkins_test()
    else f"dmpv2://mono/data/{args.workflow_task_name}/"
)
class_info_list = [
    ClassInfo(
        {
            "point_num": 5,
            "attribute_num": 0,
            "class_name": "ImageBorder",
            "attribute_names": [],
            "bbox_border_id": [0, 1, 2, 3],
        }
    )
]


# default parsing config
label_map_output_dir = f"{output_dir}/gt/anno_{args.workflow_task_name}"
label_map_config = Config.load_file(task_configs.parsing_label_map)
src_label = label_map_config.src_label
dst_label = label_map_config.dst_label
dst_label_map = (
    label_map_config.dst_label_map
    if "dst_label_map" in label_map_config.keys()
    else None
)
color_map = label_map_config.color_map
class_config = (
    label_map_config.class_config
    if "class_config" in label_map_config.keys()
    else None
)

colors, clsnames = get_colors_and_class_names_for_parsing(color_map)
# k=setting_name v=setting path
eval_setting_config = task_configs.eval_setting_config
evalset_config = task_configs.evalset_config
# generate parsing_label_name_map, id to name.
parsing_label_name_map = defaultdict(list)
[parsing_label_name_map[lid].append(lname) for lname, lid in dst_label.items()]
eval_stat_name_map = {
    k: "--".join(v) for k, v in parsing_label_name_map.items()
}


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
    if args.workflow_purpose in ["train"]:
        config["post_processor_cfgs"].append(
            dict(
                type="CacheCleanUp",
                output_dir=output_dir,
                path2clean=[os.path.join(output_dir, "gt")],
            )
        )
        config["post_processor_cfgs"].append(
            dict(
                type="ParsingAttachSource",
            )
        )
    return config


data_pack_post_processer_config = get_postprocess_config()
# ---------------------------------------------------------------------------------
#  make input configs
# ---------------------------------------------------------------------------------
# merge info
merge_config = [
    dict(
        merge_id_src=6,
        merge_id_dst=4,
        merge_iou_thresh=0.8,
        merge_pixel_thresh=1000,
    )
]
# data packer config default
default_data_packer_config = dict(
    type="LegacyDenseboxDataPacker",
    root_dir="./",
    output_densebox_rec_path=output_rec_path,
    output_densebox_json_path=output_anno_path,
    num_workers=num_worker,
)
# anno_transform default
default_annotation_ts_config = dict(
    type="Compose",
    transformer=[
        dict(
            type="DefaultGenerateLabelMapAnnoTs",
            output_dir=label_map_output_dir,
            src_label=src_label,
            dst_label=dst_label,
            reuse_prelabel=task_configs.load_prelabel,
            colors=colors,
            clsnames=clsnames,
            anno_to_contours_fn=default_anno_to_contours_fn,
            is_merge=task_configs.is_merge,
            merge_config=merge_config,
        ),
        dict(
            type="DenseBoxSegAnnoTs",
            class_ids=list(range(1, 20)),
            verify_image=True,
            verify_label=True,
            add_label_path_to_record=True,
        ),
    ],
)
# default prelabel configs
default_prelabel_configs = (
    dict(
        name="prelabel_8c_for_pack",
        output_dmp_dir=f"dmpv2://auto_tmp_2/prelabel/user/{username}/{workflow_pipeline_name}/prelabel_8c",  # noqa
        gpu_ids=(0,) if is_jenkins_test() else (0, 1, 2, 3),
        visual=True,
        visual_detection_threshold=0.3,
        num_worker=1 if is_jenkins_test() else 4,
        machines=1,
        backend="concurrent_mp_spawn",
        tags=["prelabel_8c"],
        infer_lib=InferLib.ADAS_INFER,
        config=load_yaml_with_include_and_base(
            os.path.join(
                dirname,
                "../../../pilot/densebox_pack_upload_viz_auto_matrix/configs/default_parsing/pilot_mmseg_parsing_big_model_8c_prelabel/prelabel.yaml",  # noqa
            )
        ),
        job_param=dict(
            docker_image=args.workflow_gpu_docker,
            queue_name=args.workflow_gpu_queue_name,
            walltime=60 * 24 * 14,  # one day default
            readonly_buckets=["auto_image", "auto_image_2", "adas"],
            writeable_buckets=["auto_tmp_2", "auto_tmp", "mono_aidi"],
        ),
    )
    if task_configs.is_merge
    else None
)


default_viz_dataset_config = dict(
    type="LegacyDenseBoxImageRecordDataset",
    rec_path=output_rec_path,
    anno_path=output_anno_path,
    read_only=False,
    with_seg_label=True,
    to_rgb=True,
    seg_label_dtype=np.uint8,
    as_nd=False,
)

default_viz_fn_config = dict(
    type="DenseBoxAnnoVisualizer",
    task_type="parsing",
    save_path=viz_result_path,
    parsing_color_map_config=task_configs.parsing_label_map,
)


prelabel_config = (
    default_prelabel_configs
    if task_configs.prelabel_config_update_fn is None
    else task_configs.prelabel_config_update_fn(
        default_prelabel_configs,
        **{k: v for k, v in vars().items() if not k.startswith("__")},
    )
)

local_inputs = dict(
    num_worker=num_worker,
    output_dir=output_dir,
    label_map_output_dir=label_map_output_dir,
    dmp_root_target_dir=dmp_root_target_dir,
    desc=args.workflow_description,
    group_tags=args.workflow_tags,
    group_kv_tags=dict(
        task=args.workflow_task_name,
        category=args.workflow_task_name,
        purpose=args.workflow_purpose,
    ),
    max_length=10
    if args.workflow_pipeline_test or is_jenkins_test()
    else None,
    annoset_name=args.workflow_annoset_name,
    shuffle=False,
    to_account=list(
        set([f"{username}@horizon.ai"] + task_configs.task_owners_email)
    ),
    subject=f"打包数据版本发布:{args.workflow_annoset_name}",
    # pack config
    use_roilist=False,
    class_info_list=class_info_list,
    data_packer_config=default_data_packer_config
    if task_configs.data_pack_update_fn is None
    else task_configs.data_pack_update_fn(
        default_data_packer_config,
        {k: v for k, v in vars().items() if not k.startswith("__")},
    ),
    anno_transformer_config=default_annotation_ts_config
    if task_configs.anno_transformer_update_fn is None
    else task_configs.anno_transformer_update_fn(
        default_annotation_ts_config,
        **{k: v for k, v in vars().items() if not k.startswith("__")},
    ),
    data_pack_post_processer_config=data_pack_post_processer_config,
    # create evalset config
    eval_setting_config=eval_setting_config,
    evalset_config=evalset_config,
    # datasets
    pass_invalid=args.workflow_pass_invalid,
    dp_image_meta_keys=task_configs.dp_image_meta_keys,
    dp_pack_meta_keys=task_configs.dp_pack_meta_keys,
    statistic_json_path=output_anno_path,
    eval_stat_name_map=eval_stat_name_map,
    # misc
    keep_duplicated=task_configs.keep_duplicate,
    # viz
    viz_result_path=viz_result_path,
    viz_num_show=args.viz_num_show,
    viz_dataset_config=default_viz_dataset_config
    if task_configs.viz_dataset_config_update_fn is None
    else task_configs.viz_dataset_config_update_fn(
        default_viz_dataset_config,
        **{k: v for k, v in vars().items() if not k.startswith("__")},
    ),
    viz_fn_config=default_viz_fn_config
    if task_configs.viz_visualizer_config_update_fn is None
    else task_configs.viz_visualizer_config_update_fn(
        default_viz_fn_config,
        **{k: v for k, v in vars().items() if not k.startswith("__")},
    ),
)
anno_data_manager = Constant(anno_data_manager)
remote_inputs = copy.copy(local_inputs)
traced_inputs = copy.copy(local_inputs)
traced_inputs = dict_values_to_variable(traced_inputs)


# ---------------------------------------------------------------------------------
#  build graph
# ---------------------------------------------------------------------------------
def build_graph(imperative):
    with context_resource_manager, GraphTracer(imperative=imperative):
        folder_anno_pair_list = source_data_generator(
            traced_inputs,
            anno_data_manager,
            pass_invalid=traced_inputs["pass_invalid"],
            local_root_dirs=args.workflow_source_data_root_url,
        )
        # update and merge tags
        tags = anno_data_manager.merge_tags(
            group_tags=traced_inputs["group_tags"],
            group_kv_tags=traced_inputs["group_kv_tags"],
            member_tags=traced_inputs.get("member_tags", None),
            member_kv_tags=traced_inputs.get("member_kv_tags", None),
        )
        # create annoset
        annoset = anno_data_manager.create_annoset(
            dataset_ids=[1234],
            annoset_name=traced_inputs["annoset_name"],
            group_tags=tags["group_tags"],
            group_kv_tags=tags["group_kv_tags"],
            member_tags=tags["member_tags"],
            member_kv_tags=tags["member_kv_tags"],
            desc=traced_inputs["desc"],
        )
        annoset = aidi_job(
            annoset,
            job_param=AIDIJobParam(
                task_id=append_locs(
                    "create_annoset", args.workflow_camera_loc_groups
                ),
                task_type="filter",
                docker_image=args.workflow_cpu_docker,
                queue_name=args.workflow_cpu_queue_name,
                cpu=8,
                mem_per_cpu=2,
                walltime=60,
                readonly_buckets=["auto_image", "auto_image_2"],
                writeable_buckets=["auto_tmp_2", "auto_tmp", "mono_aidi"],
            ),
        )
        data_generator = get_image_ann_pair_generator(
            folder_anno_pair_list=folder_anno_pair_list,
            annoset_ids=annoset.id,
            max_len=traced_inputs["max_length"],
        )
        additional_info_list = []
        local_perception_output_dir = to_unique_local_dir(
            prelabel_config["output_dmp_dir"]
        )
        if task_configs.is_merge:
            prelabel_op = parallel_prelabel_and_write_json(
                common_dataiter=data_generator,
                model_config=prelabel_config["config"],
                gpu_ids=prelabel_config["gpu_ids"],
                output_dir=local_perception_output_dir,
                prelabel_dir=local_perception_output_dir,
                visual=True,
                visual_detection_threshold=0.5,
                num_worker=prelabel_config["num_worker"],
                backend=prelabel_config["backend"],
                infer_lib=prelabel_config["infer_lib"],
            )
            with ControlFlow(wait=prelabel_op):
                pred_dataiter = get_simple_json_dataiter_form_dir(
                    local_perception_output_dir
                )
                additional_info_list.append(
                    get_parsing_merge_info(
                        common_dataiter=pred_dataiter,
                    )
                )
        extra_imports = import_plugin_configs(config_root)
        workflow_tails = [extra_imports]
        data_generator = get_image_ann_pair_generator(
            folder_anno_pair_list=folder_anno_pair_list,
            annoset_ids=annoset.id,
            max_len=traced_inputs["max_length"],
        )
        pack_info = legacy_densebox_pack_and_upload(
            annoset_query_str=None,
            annoset_dataiter=data_generator,
            anno_transformer_config=traced_inputs["anno_transformer_config"],
            class_info_list=traced_inputs["class_info_list"],
            data_packer_config=traced_inputs["data_packer_config"],
            output_dir=traced_inputs["output_dir"],
            num_worker=traced_inputs["num_worker"],
            dmp_root_target_dir=dmp_root_target_dir,
            annoset_name=traced_inputs["annoset_name"],
            tags=tags["group_tags"],
            kv_tags=tags["group_kv_tags"],
            member_tags=tags["member_tags"],
            member_kv_tags=tags["member_kv_tags"],
            desc=traced_inputs["desc"],
            max_length=traced_inputs["max_length"],
            shuffle=traced_inputs["shuffle"],
            use_roilist=traced_inputs["use_roilist"],
            additional_info_list=additional_info_list,
            data_pack_post_processer_config=(
                traced_inputs["data_pack_post_processer_config"]
            ),
            keep_duplicated=traced_inputs["keep_duplicated"],
        )
        if args.workflow_purpose in ["train"]:
            if GraphTracer.is_active():
                pack_info.after(extra_imports)
            statistic_infos = statistic_rec(
                pack_info,
                dp_image_meta_keys=traced_inputs["dp_image_meta_keys"],
                dp_pack_meta_keys=traced_inputs["dp_pack_meta_keys"],
                anno_path=traced_inputs["statistic_json_path"],
                temp_data_cache_dir=output_dir,
            )
            workflow_tails.append(statistic_infos)

            upload_status = anno_data_manager.update_annoset(
                annoset_id=annoset.id,
                train_pack_url=pack_info.get("url", None),
            )
            workflow_tails.append(upload_status)

            if args.workflow_viz_result:
                image_viewer_url = viz_densebox_packed_data_and_show(
                    pack_info=pack_info,
                    output_image_dir=traced_inputs["viz_result_path"],
                    viz_dataset_config=traced_inputs["viz_dataset_config"],
                    viz_fn_config=traced_inputs["viz_fn_config"],
                    viz_num_show=traced_inputs["viz_num_show"],
                )
                image_viewer_url.after(upload_status)
                workflow_tails.append(image_viewer_url)

            if not is_jenkins_test():
                send_mail = send_mail_about_annoset(
                    annoset,
                    traced_inputs["to_account"],
                    traced_inputs["subject"],
                    viz_url=None,
                )
                send_mail.after(upload_status)
                workflow_tails.append(send_mail)
            workflow_tails = aidi_job(
                workflow_tails,
                job_param=AIDIJobParam(
                    task_id=append_locs(
                        "densebox_pack_rec_to_pbrec_default_parsing",
                        args.workflow_camera_loc_groups,
                    ),
                    task_type="packing",
                    docker_image=args.workflow_cpu_docker,
                    queue_name=args.workflow_cpu_queue_name,
                    cpu=8,
                    gpu=0,
                    mem_per_cpu=8,
                    walltime=60 * 24 * 7,
                    readonly_buckets=[
                        "auto_image",
                        "auto_image_2",
                    ],
                    writeable_buckets=[
                        "auto_tmp_2",
                        "matrix",
                        "mono",
                        "auto_tmp",
                        "mono_aidi",
                    ],
                ),
            )
        elif args.workflow_purpose in ["eval"]:
            eval_set_id = upload_evalset_and_setting(
                annoset_dataiter=data_generator,
                evalset_config=traced_inputs["evalset_config"],
                setting_config=traced_inputs["eval_setting_config"],
                cache_data_url=traced_inputs["label_map_output_dir"],
                camera_loc=args.workflow_camera_loc_groups,
            )
            if GraphTracer.is_active():
                eval_set_id.after(extra_imports)
                eval_set_id.after(pack_info)
            workflow_tails.append(pack_info)

            statistic_infos = statistic_evalset(
                eval_set_id,
                temp_data_cache_dir=output_dir,
                upload_url=traced_inputs["dmp_root_target_dir"],
                dp_image_meta_keys=traced_inputs["dp_image_meta_keys"],
                dp_pack_meta_keys=traced_inputs["dp_pack_meta_keys"],
                parsing_label_name_map=traced_inputs["eval_stat_name_map"],
            )
            workflow_tails.append(statistic_infos)

            upload_status = anno_data_manager.update_annoset(
                annoset_id=annoset.id,
                eval_set_id=eval_set_id,
            )
            workflow_tails.append(upload_status)

            if not is_jenkins_test():
                send_mail = send_mail_about_annoset(
                    annoset,
                    traced_inputs["to_account"],
                    traced_inputs["subject"],
                    viz_url=None,
                )
                send_mail.after(upload_status)
                workflow_tails.append(send_mail)
            workflow_tails = aidi_job(
                workflow_tails,
                job_param=AIDIJobParam(
                    task_id=append_locs(
                        "upload_evalset_and_setting",
                        args.workflow_camera_loc_groups,
                    ),
                    task_type="filter",
                    docker_image=args.workflow_cpu_docker,
                    queue_name=args.workflow_cpu_queue_name,
                    cpu=8,
                    mem_per_cpu=4,
                    walltime=60 * 24 * 7,
                    readonly_buckets=[
                        "auto_image",
                        "auto_image_2",
                        "auto_eval",
                    ],
                    writeable_buckets=[
                        "auto_tmp_2",
                        "adas",
                        "auto_tmp",
                        "mono_aidi",
                    ],
                ),
            )
        else:
            raise ValueError(
                f"Invalid Purpose: {args.workflow_purpose}, "
                f"all valid perpuse: [train, eval]"
            )

        workflow = get_traced_graph(workflow_tails)
        return workflow


if __name__ == "__main__":
    workflow = build_graph(True)
else:
    workflow = build_graph(False)
