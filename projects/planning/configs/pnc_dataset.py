# Copyright (c) Horizon Robotics. All rights reserved.
from hat.data.transforms.planning import gt_path_func_for_data_pipeline_p3c

PATH_PREFIX = "14_pnc_data/02_user/bikun.wang/p3c"
P3C_PATH_PREFIX = "14_pnc_data/04_imitation_data"

P3C_DATASET = {
    "p3c_mini": {
        "bucket": "SD_Algorithm",
        "obs_dir": f"{PATH_PREFIX}/mini_data",
        "path_token_file": f"{PATH_PREFIX}/mini_data/path_token.yaml",
        "train": "concat_eval_intersection_left.pkl",
        "val": [
            "concat_eval_intersection_left.pkl",
            "concat_eval_intersection_right.pkl",
            "concat_eval_intersection_straight.pkl",
            "concat_eval_intersection_uturn.pkl",
            "concat_eval_straight_forward.pkl",
        ],
        "test": [
            "concat_eval_intersection_left.pkl",
            "concat_eval_intersection_right.pkl",
        ],
        "map_path_func": gt_path_func_for_data_pipeline_p3c,
    },
    "p3c_demo": {
        "bucket": "SD_Algorithm",
        "obs_dir": f"{PATH_PREFIX}/demo_7w",
        "path_token_file": f"{PATH_PREFIX}/demo_7w/path_token.yaml",
        "train": "concat_train_7w.pkl",
        "val": ["concat_val.pkl"],
        "test": ["concat_val.pkl"],
        "map_path_func": gt_path_func_for_data_pipeline_p3c,
    },
    "p3c_16w": {
        "bucket": "SD_Algorithm",
        "obs_dir": f"{P3C_PATH_PREFIX}/dataset_p3c_concat/FuncCar+Colle_2023_5_6/hat_version",  # noqa
        "path_token_file": f"{P3C_PATH_PREFIX}/dataset_p3c_concat/FuncCar+Colle_2023_5_6/hat_version/path_token.yaml",  # noqa
        "train": "concat_train_16w.pkl",
        "val": [
            "concat_eval_8w.pkl",
        ],
        "test": [
            "eval_intersection_left.pkl",
            "eval_intersection_right.pkl",
            "eval_intersection_straight.pkl",
            "eval_intersection_uturn.pkl",
            "eval_straight_forward.pkl",
        ],
        "map_path_func": gt_path_func_for_data_pipeline_p3c,
    },
    "p3c_22w": {
        "bucket": "SD_Algorithm",
        "obs_dir": f"{P3C_PATH_PREFIX}/dataset_p3c_concat/until_202307/",
        "path_token_file": f"{P3C_PATH_PREFIX}/dataset_p3c_concat/until_202307/path_token.yaml",  # noqa
        "train": "concat_train_22w.pkl",
        "val": [
            "concat_eval_8w.pkl",
        ],
        "test": [
            "eval_intersection_left.pkl",
            "eval_intersection_right.pkl",
            "eval_intersection_straight.pkl",
            "eval_intersection_uturn.pkl",
            "eval_straight_forward.pkl",
        ],
        "map_path_func": gt_path_func_for_data_pipeline_p3c,
    },
}
