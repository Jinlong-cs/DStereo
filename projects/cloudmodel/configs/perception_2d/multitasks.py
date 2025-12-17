# This script defines tasks that are supported in the multitask model
import os

__all__ = ["TASKS"]

# Tasks that have been implemented
tasks_done = [
    "detection.vehicle_detection",
    # "detection.cyclist_detection",
    # "detection.rear_detection",
    # "detection.person_detection",
    # "detection.traffic_sign_detection",
    # "detection.traffic_cone_detection",
    # "detection.traffic_light_detection",
    # "detection.road_arrow_detection",
    # "detection.vehicle_wheel_detection",
    # "detection.person_head_detection",
    # "detection.person_face_detection",
    # "detection.vehicle_light_detection",
    # "detection.cyclist_wheel_detection",
    # "detection.traffic_light_len_detection",
    # "detection.cycle_detection",
    # "detection.adb_light_detection",
    # "detection.vehicle_side_detection",
    # # the five tasks below are required by the SD project
    # "detection.cone_detection",
    # "detection.traffic_bollard_detection",
    # "detection.isolation_bollard_detection",
    # "detection.crash_barrel_detection",
    # "detection.aframe_sign_detection",
    # "segmentation.semantic_parsing_33cls_segmentation",
    # "segmentation.semantic_parsing_40cls_segmentation",
    # "segmentation.lane_instanceseg",
    # "classification.vehicle_attribute_classification",
    # "classification.vehicle_rear_attribute_classification",
    # "classification.person_attribute_classification",
    # "classification.traffic_light_primary_attribute_classification",
    # "classification.traffic_light_secondary_attribute_classification",
    # "classification.traffic_cone_attribute_classification",
    # "classification.road_arrow_attribute_classification",
    # "classification.traffic_sign_medium_attribute_classification",
    # "classification.cyclist_wheel_attribute_classification",
    # "classification.vehicle_light_detection_attribute_classification",
    # "classification.wheel_attribute_classification",
    # "classification.face_attribute_classification",
    # "classification.person_head_attribute_classification",
    # "classification.vehicle_light_recognition_classification",
    # "classification.traffic_sign_recognition_classification",
    # "classification.vehicle_negative_recognition_classification",
    # "classification.person_orientation_recognition_classification",
]

tasks = os.environ.get("CLOUDMODEL_PERCEPTION2D_TASKS", None)
if tasks is None:
    tasks = tasks_done
else:
    tasks = tasks.split(",")
folder_name = "tasks"
# TASKS includes those tasks that will be performed in multitask model
TASKS = [f"{folder_name}.{t}" for t in tasks]
