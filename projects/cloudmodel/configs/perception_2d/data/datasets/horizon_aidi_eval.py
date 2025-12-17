import copy
import os

from easydict import EasyDict

from projects.cloudmodel.configs.perception_2d.common import bucket_root

__all__ = ["mono_aidi_eval_dataset_paths", "pilot_aidi_eval_dataset_paths"]

local_datapath = os.path.join(
    bucket_root, "auto_eval/adas_eval/eval_platform/fs/"
)

# dataset ids should be in a list
task2dataset_id_mono = EasyDict(
    dict(
        person=[6029315],
        cyclist=[6029316],
        rear=[6028397],
        vehicle=[6028372],  # [6026770],  # (2160,3840)
        traffic_sign=[6028564],  # 6026233,  # (2160,3840)
        traffic_light=[6028502],
        traffic_cone=[6029527],  # 6029526,6027251,6029527,
        road_arrow=[6028431],  # 6028876, 6028431,
        semantic_parsing_33cls=[6027718, 6027717],
        semantic_parsing_40cls=[],  # (2160, 3840)
        lane_parsing_4cls=[6027559],  # (2160, 3840)
        lane_parsing_6cls=[6028550],  # (2160, 3840)
        lane=[],
        # sd cone
        cone=[6029527],  # [6040436]
        traffic_bollard=[6029527],
        isolation_bollard=[6029527],
        crash_barrel=[6029527],
        aframe_sign=[6041008],
        # new tasks
        vehicle_wheel=[6040778],
        person_head=[6036291],
        person_face=[6040762, 6041137],
        cyclist_wheel=[6027361],
        traffic_light_len=[6028514],
        cycle=[6040964],
        vehicle_plate=[6040434, 6040435],
        vehicle_side=[6041275],  # 6041275, 6040601, 6040553, 6042266
        adb_light=[6041719],
    )
)

task2dataset_id_pilot = EasyDict(
    dict(
        # C385
        # vehicle=[6036083, 6036735, 6028646, 6029412, 6029418],
        # AS33
        vehicle=[
            6027456,
            6029359,
            6026506,
            6026847,
            6029386,
            6028855,
            6028996,
            6029415,
        ],
        # AS33
        cyclist=[6026851, 6028984, 6026388, 6026951, 6027190],
        # AS33
        rear=[6027121, 6028916, 6027199, 6027111, 6027003],
        # AS33
        person=[
            6027363,
            6028666,
            6029356,
            6029088,
            6029101,
            6028981,
            6026368,
            6026947,
        ],
        semantic_parsing_33cls=[
            6027718,
            6027717,
            6028089,
            6028088,
            6028087,
            6028086,
            6028085,
        ],
        cone=[],
        traffic_bollard=[],
        isolation_bollard=[],
        crash_barrel=[],
        aframe_sign=[],
        vehicle_wheel=[6029565, 6026789, 6029463],
        person_face=[6037601, 6034106],
    )
)

aidi_eval_task2id_list = {
    "mono": task2dataset_id_mono,
    "pilot": task2dataset_id_pilot,
}

# To support evaluating multiple datasets at a time on AIDI platform, we
# construct the AIDI eval data path in the structure below:
# {
#     "cyclist": {
#         "6026388": "/horizon-bucket/auto_eval/adas_eval/eval_platform/fs/6026388/datasets", # noqa
#         "6026851": "/horizon-bucket/auto_eval/adas_eval/eval_platform/fs/6026388/datasets", # noqa
#     },
#     "person": {
#         "6026368": "/horizon-bucket/auto_eval/adas_eval/eval_platform/fs/6026368/datasets", # noqa
#         "6029101": "/horizon-bucket/auto_eval/adas_eval/eval_platform/fs/6029101/datasets", # noqa
#     }
#     ...
# }

for proj, task2id in aidi_eval_task2id_list.items():
    aidi_eval_dataset_paths = EasyDict(dict())
    for cls, dataset_ids in task2id.items():
        assert isinstance(dataset_ids, list), "dataset_ids must be a list"
        aidi_eval_dataset_paths.update(
            {
                cls: {
                    str(dataset_id): os.path.join(
                        local_datapath, str(dataset_id), "datasets"
                    )
                    for dataset_id in dataset_ids
                }
            }
        )
    if "mono" in proj:
        mono_aidi_eval_dataset_paths = copy.deepcopy(aidi_eval_dataset_paths)
    elif "pilot" in proj:
        pilot_aidi_eval_dataset_paths = copy.deepcopy(aidi_eval_dataset_paths)
    else:
        raise ValueError(f"Currently, don't support {proj} aidi eval!")
