import copy
import json
import os
import os.path as osp
from collections import OrderedDict

import yaml
from horizon_plugin_pytorch.march import March

from hat.core.proj_spec.classification import (
    get_mono_classification_desc,
    get_rcnn_classification_desc,
)
from hat.core.proj_spec.detection import (
    get_class_names_used_in_desc,
    get_det_default_merge_fn_type_and_params,
)

march = March.BERNOULLI2  # bayes1 bpu2.2 bayes bpu2.5
backbone_type = "TinyVargNetV2"
alpha = 0.5
unet_out_strides = [4]
_channel_list = [32, 32, 64, 128, 256, 512]
backbone_out_channels = [int(i * alpha) for i in _channel_list]
stride2channels = {2 ** i: c for i, c in enumerate(backbone_out_channels, 1)}
bn_kwargs = dict(eps=1e-5, momentum=0.1)
add_tracking = True
tracking_feature_bs = 1  # 对应编译时候得batch_size

vargnetv2_backbone = dict(
    type=backbone_type,
    num_classes=1000,  # must be assigned, but doesn't work
    bn_kwargs=bn_kwargs,
    alpha=alpha,
    group_base=8,
    include_top=False,
    extend_features=True,
    input_resize_scale=None,
    channel_list=_channel_list,
)

backbone_factory = dict(TinyVargNetV2=vargnetv2_backbone)
backbone = backbone_factory[backbone_type]

pretrain_checkpoint = "http://fm-hao-chen.ucloudtrain.hogpu.cc/plat_gpu/TinyVarGNetV2_CLS-HAT-pretrain-20220223-113926/output/models/TinyVarGNetV2_CLS/float-checkpoint-best-9b796482.pth.tar"  # noqa
backbone_neck_modules = [
    [
        "backbone",
        "person_pos_occ_classification_neck",
        "person_head_detection_neck",
    ]
]

training_step = os.getenv("HAT_TRAINING_STEP", "float")
assert training_step in ["float", "float_freeze_bn", "qat", "int_infer"]

task_loss_weights = {
    "person_head_detection": 1,
    "person_age_classification": 1,
    "person_orientation_classification": 1,
    "person_pos_occ_classification": 2 if training_step == "qat" else 1,
    "person_pose_classification": 1,
}
task_list = task_loss_weights.keys()

norm_len = 116
norm_method = "height"
input_size = (64, 128)
compile_model = training_step == "int_infer"

num_machines = 2
num_gpus_per_machine = 8
val_interval = 1000
val_interval_by = "step"
input_sequence_length = 1
test_image_dir = None
test_batch_size_per_gpu = 4
val_det_batch_size_per_gpu = 16

pipeline_test = os.getenv("HAT_PIPELINE_TEST", "0") == "1"
val_only = os.getenv("HAT_VAL_ONLY", "0") == "1"
local_train = not os.path.exists("/running_package")
if local_train:
    device_ids = [3]
    log_freq = 25
    bucket_root = "/horizon-bucket/mono/"
    save_prefix = "tmp_output"
    redirect_config_logging_path = None

    train_num_workers, val_num_workers, test_num_workers = 4, 16, 4
    train_batch_size_per_gpu = 256
    fast_debug = False
else:
    device_ids = list(range(num_gpus_per_machine))
    log_freq = 25
    bucket_root = "/bucket/input/mono/"
    save_prefix = "/job_data/models/"
    redirect_config_logging_path = (
        f"/job_log/hat_config_output_{training_step}.log"  # noqa
    )

    train_num_workers, val_num_workers, test_num_workers = 10, 1, 0
    train_batch_size_per_gpu = 128
    fast_debug = False


def append_extra_node_func(nodes, feats, mode):
    # skip this func when not build test_model or use_cat_filter=False
    if mode != "test":
        return OrderedDict()
    name2out = OrderedDict()
    det_task_names = ["person_head_detection"]
    for task_name in det_task_names:
        feats = nodes[f"{task_name}_neck"](feats)
        feats = nodes[f"{task_name}_head"](feats)
        anchor = nodes["anchor_generator"](feats["head_predict_rpn_head_out"])
        dpp_output = nodes["dpp_postprocess"](anchor, feats)
        name2out.update({task_name: dpp_output})
    return name2out


last_stride = 2 ** len(backbone_out_channels)
tracking_feature_desc = {
    "task": "tracking_feature",
    "size": [
        tracking_feature_bs,
        input_size[1] // last_stride,
        input_size[0] // last_stride,
        backbone_out_channels[-1],
    ],
}
add_tracking_feature_desc = dict(
    type="AddDesc",
    strict=True,
    per_tensor_desc=[json.dumps(tracking_feature_desc)],
)
tracking_module = dict(
    type="OutputModule",
    head=dict(type="ClassificationTrackHead"),
    prefix="head",
    postprocess=add_tracking_feature_desc,
)


adas_eval_datadet_id_list = {
    "person_head_detection": [6027154, 6025708, 6025709],
    "person_pose_classification": [6028197, 6024745, 6026670],
    "person_age_classification": [6027225],
    "person_pos_occ_classification": [6027664],
    "person_posneg_classification": [
        6027322,
        6029231,
        6027322,
        6027352,
        6027070,
        6026850,
    ],
}
local_datapath = osp.join(bucket_root, "auto_eval/adas_eval/eval_platform/fs/")
dataset_train_rect_path = [
    local_datapath + task + "/rec/" + "train.rec" for task in task_list
]  # noqa
annotations_train_path = [
    local_datapath + task + "/rec/" + "train.json" for task in task_list
]  # noqa
dataset_val_img_path = [
    os.path.join(local_datapath, str(dataset_id), "datasets")
    for dataset_id in adas_eval_datadet_id_list.values()
]  # noqa


def get_dataset(
    recs,
    pbrecs,
    rec_idxs,
    batchsize_list=None,
    dataset_type="distributed_compose_dataset",
    **kwargs,
):
    if dataset_type == "distributed_compose_dataset":
        config = dict(
            type="ComposeIterableDataset",
            datasets=[
                dict(
                    type="DenseboxDataset2PE",
                    data_path=rec_path_i,
                    anno_path=anno_path_i,
                    rec_idx_file_path=idx_path_i,
                    **kwargs,
                )
                for rec_path_i, anno_path_i, idx_path_i in zip(
                    recs, pbrecs, rec_idxs
                )  # noqa
            ],
            batchsize_list=batchsize_list,
        )
    else:
        config = dict(
            type="ConcatDataset",
            datasets=[
                dict(
                    type="DenseboxDataset2PE",
                    data_path=rec_path_i,
                    anno_path=anno_path_i,
                    rec_idx_file_path=idx_path_i,
                    **kwargs,
                )
                for rec_path_i, anno_path_i, idx_path_i in zip(
                    recs, pbrecs, rec_idxs
                )  # noqa
            ],
        )
    return config


def change_dataset(_loader, task_name, data_source, fast_debug=fast_debug):
    data_loader = copy.deepcopy(_loader)

    if data_source == "mono":
        cfg_dir = os.path.dirname(__file__)
        with open(f"{cfg_dir}/datasets.yaml", "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        data_source_inst = {}
        for task in data.keys():
            data_source_inst[task] = [
                {
                    "imgrec": f"{bucket_root}/{d['train_rec']}",
                    "imgrec_idx": f"{bucket_root}/{d['train_json'].replace('anno.pb_rec', 'rec.idx')}",  # noqa
                    "pbrec": f"{bucket_root}/{d['train_json']}",
                    "sample_weight": d["batch_size"],
                }
                for d in data[task]
            ]
    else:
        raise ValueError(
            f"only support 'mono', 'pilot', 'merge_selected_mono_into_pilot', but found {data_source}"  # noqa
        )

    def _create_mono_dataset(task_name, old_dataset):
        data_records = data_source_inst[task_name]
        if fast_debug:
            # reduce to 1 rec, only for debug use
            data_records = data_records[:1]

        recs = [item["imgrec"] for item in data_records]
        pbrecs = [item["pbrec"] for item in data_records]
        rec_idxs = [item["imgrec_idx"] for item in data_records]
        sample_weight = [item["sample_weight"] for item in data_records]

        kwargs = copy.deepcopy(old_dataset)
        kwargs.pop("type")
        kwargs.pop("data_path")
        kwargs.pop("anno_path")
        mono_dataset = get_dataset(
            recs, pbrecs, rec_idxs, sample_weight, **kwargs
        )
        return mono_dataset

    if task_name == "person_pos_occ_classification":
        data_loader["batch_size"] *= 2
    if task_name == "person_orientation_classification":
        data_loader["batch_size"] //= 4

    data_loader["dataset"] = _create_mono_dataset(
        task_name, data_loader["dataset"]
    )
    return data_loader


if compile_model is False:

    def get_classification_model_desc(
        task_name,
        output_name,
        class_name,
        desc_id,
        prediction_return_cnt=1,
    ):
        per_tensor_desc = get_rcnn_classification_desc(
            "frcnn_classification", output_name, class_name, desc_id
        )
        per_tensor_desc = json.loads(per_tensor_desc)
        per_tensor_desc.update(
            crop_desc=dict(
                norm_len=norm_len,
                norm_method=norm_method,
                image_size=input_size[::-1],
                padding=None,
            ),
        )
        classification_model_desc = dict(
            type="AddDesc",
            strict=True,
            per_tensor_desc=[
                json.dumps(per_tensor_desc)
                for _ in range(prediction_return_cnt)
            ],
        )
        return classification_model_desc

    def get_anchor_model_desc(
        classnames,
        legacy_bbox,
        task_name,
        roi_region=None,
        prediction_return_cnt=4,
        vanishing_point=None,
        image_hw=None,
        anchor_wh=(16.0, 16.0),
    ):
        (
            merge_fn_type,
            merge_fn_params,
        ) = get_det_default_merge_fn_type_and_params()
        anchor_model_desc = dict(
            type="AddDesc",
            strict=True,
            per_tensor_desc=[
                json.dumps(
                    dict(
                        task="frcnn_detection",
                        class_name=get_class_names_used_in_desc(classnames),
                        class_agnostic=True,
                        score_act_type="identity",
                        with_background=False,
                        mean=(0, 0, 0, 0),
                        std=(1, 1, 1, 1),
                        reg_type="rcnn",
                        legacy_bbox=int(
                            legacy_bbox
                        ),  # use 0/1 instead of False/True
                        nms_threshold=0.7,
                        roi_regions=roi_region,
                        vanishing_point=vanishing_point,
                        merge_fn_type=merge_fn_type,
                        merge_fn_params=merge_fn_params,
                        crop_desc=dict(
                            norm_len=norm_len,
                            norm_method=norm_method,
                            image_size=input_size[::-1],
                            padding=None,
                        ),
                    )
                )
                for _ in range(prediction_return_cnt)
            ],
        )

        return anchor_model_desc


else:

    def get_classification_model_desc(
        task_name,
        output_name,
        class_name,
        desc_id,
        prediction_return_cnt=1,
    ):
        per_tensor_desc = get_mono_classification_desc(
            task_name, output_name, class_name, desc_id
        )
        per_tensor_desc = json.loads(per_tensor_desc)
        per_tensor_desc.update(
            image_size=input_size[::-1],
            norm_method=norm_method,
            norm_len=norm_len,
        )
        classification_model_desc = dict(
            type="AddDesc",
            strict=True,
            per_tensor_desc=[
                json.dumps(per_tensor_desc)
                for _ in range(prediction_return_cnt)
            ],
        )
        return classification_model_desc

    def get_anchor_model_desc(
        classnames,
        legacy_bbox,
        task_name,
        image_hw=None,
        prediction_return_cnt=4,
        anchor_wh=(16.0, 16.0),
    ):
        anchor_model_desc = dict(
            type="AddDesc",
            strict=True,
            per_tensor_desc=[
                json.dumps(
                    dict(
                        task=task_name,
                        anchor_wh_pair=[int(v) for v in anchor_wh],
                        linear_a=4,
                        linear_b=2.0,
                        class_name=get_class_names_used_in_desc(classnames),
                        reg_type="frcnn",
                        legacy_bbox=int(legacy_bbox),
                        pixel_center_align=0,
                        score_threshold=[0.25],
                        image_size=image_hw,
                        norm_method=norm_method,
                        norm_len=norm_len,
                    )
                )
                for _ in range(prediction_return_cnt)
            ],
        )
        return anchor_model_desc


def get_anchor_post_process_cfg(
    anchor_args,
    task_name,
    *,
    use_clippings=True,
    nms_iou_threshold=0.3,
    box_filter_threshold=0.25,
    pre_nms_top_k=2000,
    post_nms_top_k=100,
    nms_margin=0.0,
    nms_padding_mode="pad_zero",
    bbox_min_hw=(1, 1),
    input_shift=4,
):
    anchor_pred = dict(
        type="AnchorPostProcess",
        num_classes=anchor_args["num_fg_classes"],
        class_offsets=[0] * len(anchor_args["feat_strides"]),
        use_clippings=use_clippings,
        image_hw=input_size[::-1],
        nms_iou_threshold=nms_iou_threshold,
        pre_nms_top_k=pre_nms_top_k,
        post_nms_top_k=post_nms_top_k,
        nms_margin=nms_margin,
        box_filter_threshold=box_filter_threshold,
        input_key="head_predict_rpn_head_out",
        nms_padding_mode=nms_padding_mode,
        bbox_min_hw=bbox_min_hw,
        input_shift=input_shift,
    )
    return anchor_pred
