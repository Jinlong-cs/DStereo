import uuid
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Union

import numpy as np
from hatbc.message import Attribute, BBox3D, Instance, LidarFrame, MessageMeta
from sp_middle_model_configs import *  # noqa: F403

from hat.core.data_struct.app_struct import DetObjects
from hat.data.collates.collates import collate_mmfusion_3d

# import to register hat models
from hat.models.task_modules import lidar
from hat.utils.apply_func import _as_list

# -------------------------- lidar params --------------------------
pc_load_dimension = 4  # point cloud dimension to load: 6 or 4
pc_keep_dimension = 4  # point cloud dimension to use
pc_load_dtype = np.float32  # point cloud dtype to use: np.float64/np.float32
pc_range = (-2, -51.2, -2.0, 100.4, 51.2, 4.0)
voxel_size = (0.1, 0.1, 0.15)

# -------------------------- detector params --------------------------
bn_kwargs = dict(eps=1e-5, momentum=0.1)
num_class = 7
score_threshold = 0.1

ID2NAME = {
    0: "Car",
    1: "Cyclist",
    2: "Tricycle",
    3: "Pedestrian",
    4: "Construction",
    5: "Truck",
    6: "Bus",
}

# data preparation (transforms)
val_transforms = [
    dict(
        type="ParsePointCloud",
        dtype=pc_load_dtype,
        load_dim=pc_load_dimension,
        keep_dim=pc_keep_dimension,
    ),
    dict(
        type="Point2VCS",
        shuffle_points=False,
    ),
    dict(
        type="Voxelization",
        range=pc_range,
        voxel_size=voxel_size,
        max_points_in_voxel=15,
        max_voxel_num=500000,
    ),
]


# network modules
point_cloud_encoder = dict(
    type="MeanVFE",
    num_input_features=4,
)

SpModel = "SpMiddleResNetD4_lite112"
lidar_backbone_model_cfg = eval(SpModel)
lidar_backbone = dict(
    type="SpMiddleModel",
    num_input_features=4,
    model_cfg=lidar_backbone_model_cfg,
    feature_transpose=True,
)

lidar_neck = dict(
    type="ASPPNeck",
    in_channels=256,
    block_nums=3,
)

bbox_head = dict(
    type="AfdetHead",
    in_channels=256,
    bn_kwargs=bn_kwargs,
    num_class=num_class,
    common_heads={
        "reg": (2, 2),
        "height": (1, 2),
        "dim": (3, 2),
        "rot": (2, 2),
    },  # (output_channel, num_conv)
    share_conv_channel=48,
    split_hm_head=False,
)

max_pool_kernel = [3, 3, 3, 3, 3, 3, 3]
detection_alpha = [2.4, 1.8, 2.4, 3.4, 2.0, 1.6, 0.8]
maxpool_dict = dict(
    maxpool_kernel_size=[max_pool_kernel], topk=[200] * num_class
)
predict_det = dict(
    type="AfdetPredict",
    num_classes=[num_class],
    maxpool_dict=maxpool_dict,
    use_bev_format=False,  # bev eval metric
)


def topo_builder(nodes, _inputs, task_name):
    name2out = OrderedDict()

    voxel_feats = nodes["point_cloud_encoder"](
        _inputs["voxel_data"][0],
        _inputs["voxel_num_points"][0],
    )

    lidar_backbone_out = nodes["lidar_backbone"](
        voxel_feats,
        _inputs["voxel_coordinates"][0],
        _inputs["num_voxels"][0].shape[0],
        _inputs["voxel_shape"][0],
    )

    lidar_feats, _ = nodes["lidar_neck"](lidar_backbone_out)

    preds = nodes["lidar_bbox_head"](lidar_feats)

    ret = [
        nodes["lidar_predict_det"](
            _inputs,
            preds,
            pc_range,
            [detection_alpha],
        )
    ]

    name2out[task_name] = ret
    return name2out


def get_topo_builder(task_name):
    def _build_topo(nodes, inputs):
        name2out = OrderedDict()

        name2out.update(
            topo_builder(
                nodes=nodes,
                _inputs=inputs,
                task_name=task_name,
            )
        )

        return name2out

    return _build_topo


def get_matrix_from_frame(frame: LidarFrame) -> np.ndarray:
    """
    从 LidarFrame 中获取 Lidar 外参 4 x 4 变换矩阵.
    """

    def _euler_to_rotation_matrix(yaw, pitch, roll):
        R_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)],
            ]
        )
        R_y = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)],
            ]
        )
        R_z = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1],
            ]
        )
        matrix = np.dot(R_z, np.dot(R_y, R_x))
        return matrix

    lidar_param = frame.lidar_param

    matrix = np.zeros([4, 4])
    matrix[:3, :3] = _euler_to_rotation_matrix(
        lidar_param.yaw, lidar_param.pitch, lidar_param.roll
    )
    matrix[:3, -1] = [
        lidar_param.x_offset,
        lidar_param.y_offset,
        lidar_param.z_offset,
    ]
    matrix[-1, -1] = 1.0
    matrix = matrix.astype(np.float32)

    return matrix


def lidar_frame_to_hat_dict(msg: LidarFrame):
    """Convert lidar frame to hat dict.

    Args:
        msg: lidar frame.
    """
    point_cloud = msg.pcl.data
    pc_meta = dict(
        point_clouds=[point_cloud],
        meta_info=dict(
            T_lidar2vcs=get_matrix_from_frame(msg),
        ),
        object_token=msg.pcl.url,
    )
    return pc_meta


def preprocess(
    data: Union[
        LidarFrame,
        List[LidarFrame],
    ],
    transforms: List[Callable],
) -> Dict:
    def _transform(frame):
        pc_meta = lidar_frame_to_hat_dict(frame)
        for t in transforms:
            pc_meta = t(pc_meta)
        pc_meta.pop("meta_info")
        pc_meta.pop("point_clouds")
        return pc_meta

    data = _as_list(data)
    pc_metas = list(map(_transform, data))
    batch_data = collate_mmfusion_3d(pc_metas)
    return batch_data


def get_inference_models(task_name):
    inputs = dict(
        voxel_data=None,
        voxel_shape=None,
        voxel_num_points=None,
        num_voxels=None,
        voxel_coordinates=None,
        object_token=None,
    )

    nodes = dict(
        point_cloud_encoder=point_cloud_encoder,
        lidar_backbone=lidar_backbone,
        lidar_neck=lidar_neck,
        lidar_bbox_head=bbox_head,
        lidar_predict_det=predict_det,
    )

    val_model = dict(
        type="GraphModel",
        nodes=nodes,
        inputs=inputs,
        topology_builder=get_topo_builder(task_name),
        lazy_forward=True,
    )

    return val_model


def postprocess(
    preds: Dict[str, List[DetObjects]],
    data: Dict[str, Any],
    task_names,
) -> List[List[Instance]]:
    """
    Convert model predictions to list of instances from each frames.
    """

    preds_dict = preds[0]._asdict()  # namedtuple to dict
    num_frames = 0
    for key in preds_dict.keys():
        if "token" in key:
            num_frames += 1

    pred_results = []

    for frame_idx in range(num_frames):
        frame_results = []

        box3d_key = task_names[0] + "_" + str(frame_idx) + "_box3d_lidar"
        scores_key = task_names[0] + "_" + str(frame_idx) + "_scores"
        label_key = task_names[0] + "_" + str(frame_idx) + "_label_preds"
        token_key = task_names[0] + "_" + str(frame_idx) + "_token"

        timestamp = preds_dict[token_key].split("/")[-1].split(".")[0]

        inst_meta = MessageMeta(
            # channel=DataModuleID.DATA_MODULE_ID_LIDAR_FRONT,  # AT128, channel=31
            timestamp=int(timestamp),
        )

        for box3d, score, label in zip(
            preds_dict[box3d_key],
            preds_dict[scores_key],
            preds_dict[label_key],
        ):
            box3d = box3d.cpu().numpy()
            score = score.cpu().numpy()
            label = int(label.cpu())

            if score < score_threshold:
                continue

            bbox3ds = [
                BBox3D(
                    score=score,
                    loc=box3d[:3],
                    # dim: w/h/l
                    dim=[box3d[3], box3d[5], box3d[4]],
                    yaw=box3d[6],
                )
            ]
            inst = Instance(
                meta=inst_meta,
                topic=ID2NAME[label],
                bbox3ds=bbox3ds,
                attributes=[Attribute(topic="category", value=ID2NAME[label])],
            )
            frame_results.append(inst)

        pred_results.append(frame_results)

    return pred_results, data
