import os

from easydict import EasyDict

from tests import HAT_BUCKET_PATH

################################
# Only used for automated test #
################################
test_level = os.environ.get("HAT_PILOT_TEST_LEVEL")

if test_level == "commit":
    # use packed data in bucket
    root = HAT_BUCKET_PATH
    if not os.path.isdir(root):
        raise FileNotFoundError("HDLTAlgorithm bucket")
    root = os.path.join(root, "users/yilin.xiong/pilot/data/test_unit")
elif test_level == "daily":
    # use freshly packed data in local
    root = f"{os.path.dirname(__file__)}/../../pack_tools/data/lmdb"
else:
    raise ValueError(f"Get unknown test_level: {test_level}")

datapaths = dict(
    vehicle_3d_detection=dict(
        dict(
            train_data_paths=[
                dict(
                    data_path=[f"{root}/vehicle_3d_detection"],  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    person_3d_detection=dict(
        dict(
            train_data_paths=[
                dict(
                    data_path=[f"{root}/ped_3d_detection"],  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    vehicle=dict(
        dict(
            train_data_paths=[
                dict(
                    data_path=f"{root}/vehicle_detection",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    cyclist=dict(
        dict(
            train_data_paths=[
                dict(
                    data_path=f"{root}/cyclist_detection",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    person=dict(
        dict(
            train_data_paths=[
                dict(
                    data_path=f"{root}/person_detection",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    semantic_parsing=dict(
        dict(
            train_data_paths=[
                dict(
                    data_path=f"{root}/default_parsing",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    lane_parsing=dict(
        dict(
            train_data_paths=[
                dict(
                    data_path=f"{root}/lane_parsing",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    person_face_detection=dict(
        dict(
            train_data_paths=[
                dict(
                    data_path=f"{root}/person_face_detection",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    person_occlusion_classification=dict(
        dict(
            train_data_paths=[
                dict(
                    data_path=f"{root}/person_occlusion_classification",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    person_orientation_classification=dict(
        dict(
            train_data_paths=[
                dict(
                    data_path=f"{root}/person_orientation_classification",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    person_pose_classification=dict(
        dict(
            train_data_paths=[
                dict(
                    data_path=f"{root}/person_pose_classification",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    rear=dict(
        dict(
            train_data_paths=[
                dict(
                    data_path=f"{root}/rear_detection",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    rear_occlusion_classification=dict(
        dict(
            train_data_paths=[
                dict(
                    data_path=f"{root}/rear_occlusion_classification",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    rear_part_classification=dict(
        dict(
            train_data_paths=[
                dict(
                    data_path=f"{root}/rear_part_classification",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    rear_plate_detection=dict(
        dict(
            train_data_paths=[
                dict(
                    data_path=f"{root}/rear_plate_detection",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    vehicle_category=dict(
        dict(
            train_data_paths=[
                dict(
                    data_path=f"{root}/vehicle_category_classification",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    vehicle_ground_line=dict(
        dict(
            train_data_paths=[
                dict(
                    data_path=f"{root}/vehicle_ground_line",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    vehicle_occlusion_classification=dict(
        dict(
            train_data_paths=[
                dict(
                    data_path=f"{root}/vehicle_occlusion_classification",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    vehicle_truncation_classification=dict(
        dict(
            train_data_paths=[
                dict(
                    data_path=f"{root}/vehicle_truncation_classification",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    vehicle_wheel_detection=dict(
        dict(
            train_data_paths=[
                dict(
                    data_path=f"{root}/vehicle_wheel_detection",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    vehicle_wheel_kps=dict(
        dict(
            train_data_paths=[
                dict(
                    data_path=f"{root}/vehicle_wheel_kps",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    # using exist data
    vehicle_flank=dict(
        dict(
            train_data_paths=[
                dict(
                    data_path=f"{HAT_BUCKET_PATH}/users/yilin.xiong/pilot/data/test_unit/vehicle_flank",
                    sample_weight=1,
                ),
            ],
        ),
    ),
    image_fail_parsing=dict(
        dict(
            train_data_paths=[
                dict(
                    data_path=f"{root}/image_fail_parsing",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
)

datapaths = EasyDict(datapaths)
