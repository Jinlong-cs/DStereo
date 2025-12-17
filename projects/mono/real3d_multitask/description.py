import json
import math

global_desc = dict(
    roi_input=dict(
        fp_x=480,
        fp_y=50,
        width=960,
        height=192,
    ),
    vanishing_point=[480, 270],
)


def vehicle_detection_desc(
    focal_length_default,
    undistort_point_method="Pinhole",
    use_multibin=True,
    multibin_centers=(0.0, math.pi / 2, math.pi, -math.pi / 2),
):

    per_tensor_desc = [
        {
            "task": "camera_3d_detection",
            "output_name": "heatmap_output",
            "focal_length_default": focal_length_default,
            "input_resize_ratio": 1.0,
            "score_threshold": 0.3,
            "topk": 40,
            "kernel_size": [3, 3],
            "properties": [{"channel_labels": ["vehicle"]}],
            **global_desc,
        },
        {
            "task": "camera_3d_detection",
            "output_name": "depth_output",
            "properties": [{"channel_labels": ["depth"]}],
            **global_desc,
        },
        {
            "task": "camera_3d_detection",
            "output_name": "rot_output",
            "use_multibin": 0,
            "properties": [{"channel_labels": ["sin", "cos"]}],
            **global_desc,
        },
        {
            "task": "camera_3d_detection",
            "output_name": "dim_output",
            "properties": [{"channel_labels": ["height", "width", "length"]}],
            **global_desc,
        },
        {
            "task": "camera_3d_detection",
            "output_name": "loc_offset_output",
            "properties": [{"channel_labels": ["offset_x", "offset_y"]}],
            **global_desc,
        },
        {
            "task": "camera_3d_detection",
            "output_name": "wh_output",
            "properties": [{"channel_labels": ["bbox_w", "bbox_h"]}],
            **global_desc,
        },
    ]
    for idx, desc in enumerate(per_tensor_desc):
        if desc["output_name"] == "heatmap_output" and undistort_point_method:
            per_tensor_desc[idx][
                "undistort_point_method"
            ] = undistort_point_method
        if desc["output_name"] == "rot_output" and use_multibin:
            per_tensor_desc[idx]["use_multibin"] = 1
            per_tensor_desc[idx]["multibin_centers"] = multibin_centers
            per_tensor_desc[idx]["properties"] = [
                {
                    "channel_labels": [
                        f"rot{i}" for i in range(len(multibin_centers) * 3)
                    ]
                }
            ]
    per_tensor_desc = [json.dumps(i) for i in per_tensor_desc]
    return per_tensor_desc


def face_detection_desc():
    per_tensor_desc = [
        {
            "task": "centernet_detection",
            "class_name": "face",
            "output_name": "heatmap",
            "score_threshold": 0.3,
            "topk": 40,
            "kernel_size": [3, 3],
            "properties": [{"channel_labels": ["face"]}],
            **global_desc,
        },
        {
            "task": "centernet_detection",
            "class_name": "face",
            "output_name": "wh",
            "properties": [{"channel_labels": ["bbox_w", "bbox_h"]}],
            **global_desc,
        },
    ]
    per_tensor_desc = [json.dumps(i) for i in per_tensor_desc]
    return per_tensor_desc


def plate_detection_desc():
    per_tensor_desc = [
        {
            "task": "centernet_detection",
            "class_name": "plate",
            "output_name": "heatmap",
            "score_threshold": 0.3,
            "topk": 40,
            "kernel_size": [3, 3],
            "properties": [{"channel_labels": ["plate"]}],
            **global_desc,
        },
        {
            "task": "centernet_detection",
            "class_name": "plate",
            "output_name": "wh",
            "properties": [{"channel_labels": ["bbox_w", "bbox_h"]}],
            **global_desc,
        },
    ]
    per_tensor_desc = [json.dumps(i) for i in per_tensor_desc]
    return per_tensor_desc
