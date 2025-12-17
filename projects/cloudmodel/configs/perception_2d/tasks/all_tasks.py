from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

from dataclasses_json import DataClassJsonMixin
from hatbc.utils import Enum


# ------------------------------------------
# tasks
# ------------------------------------------
class ModelType(Enum):
    image_model = "image_model"
    roi_model = "roi_model"


class TaskType(Enum):
    detection = "detection"
    instanceseg = "instanceseg"
    segmentation = "segmentation"
    classification = "classification"
    keypoints = "keypoints"


@dataclass
class CloudModelTask(DataClassJsonMixin):
    task_name: str
    task_type: TaskType
    model_type: ModelType
    class_name: List[str]
    subclass_name: Optional[List[str]] = None
    class_id: Optional[List[int]] = None

    def __hash__(self):
        return hash(
            self.task_name + self.task_type.value + self.model_type.value
        )

    def __str__(self):
        return (
            f"task_name: {self.task_name}, "
            f"task_type: {self.task_type.value}, "
            f"model_type: {self.model_type.value}"
        )


class AllTasks:
    # ------------------------------------------
    # detection
    # ------------------------------------------
    person_detection = CloudModelTask(
        task_name="person_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["person"],
        class_id=[1],
    )
    traffic_light_detection = CloudModelTask(
        task_name="traffic_light_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["traffic_light"],
        class_id=[2],
    )
    traffic_light_shell_detection = CloudModelTask(
        task_name="traffic_light_shell_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["traffic_light_shell"],
        class_id=[2],
    )
    traffic_sign_detection = CloudModelTask(
        task_name="traffic_sign_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["traffic_sign"],
        class_id=[3],
    )
    vehicle_detection = CloudModelTask(
        task_name="vehicle_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["vehicle"],
        class_id=[4],
    )
    rear_detection = CloudModelTask(
        task_name="rear_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["vehicle_rear", "rear"],
        class_id=[5],
    )
    traffic_light_len_detection = CloudModelTask(
        task_name="traffic_light_len_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["traffic_light_len", "traffic_light_lens"],
        class_id=[6, 6],
    )
    cyclist_detection = CloudModelTask(
        task_name="cyclist_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["cyclist"],
        class_id=[7],
    )
    road_arrow_detection = CloudModelTask(
        task_name="road_arrow_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["road_arrow"],
        class_id=[8],
    )
    traffic_cone_detection = CloudModelTask(
        task_name="traffic_cone_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["traffic_cone"],
        class_id=[9],
    )
    parking_lock_detection = CloudModelTask(
        task_name="parking_lock_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["parking_lock"],
        class_id=[10],
    )
    person_ped_detection = CloudModelTask(
        task_name="person_ped_detection",
        task_type=TaskType.detection,
        model_type=ModelType.roi_model,
        class_name=["person"],
        class_id=[1],
    )
    cyclist_2pe_detection = CloudModelTask(
        task_name="cyclist_2pe_detection",
        task_type=TaskType.detection,
        model_type=ModelType.roi_model,
        class_name=["cyclist"],
        class_id=[7],
    )
    vehicle_plate_detection = CloudModelTask(
        task_name="vehicle_plate_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["vehicle_plate"],
        class_id=[2],
    )
    vehicle_2pe_detection = CloudModelTask(
        task_name="vehicle_2pe_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["vehicle"],
        class_id=[4],
    )
    # detection new task
    vehicle_light_detection = CloudModelTask(
        task_name="vehicle_light_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["vehicle_light"],
        class_id=[11],
    )
    vehicle_wheel_detection = CloudModelTask(
        task_name="vehicle_wheel_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["vehicle_wheel"],
        class_id=[4],
    )
    person_head_detection = CloudModelTask(
        task_name="person_head_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["person_head"],
        class_id=[4],
    )
    # TODO difference between `person_face` and `face`
    person_face_detection = CloudModelTask(
        task_name="person_face_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["person_face"],
        class_id=[10],
    )
    cyclist_wheel_detection = CloudModelTask(
        task_name="cyclist_wheel_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["cyclist_wheel"],
        class_id=[11],
    )
    cycle_detection = CloudModelTask(
        task_name="cycle_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["cycle"],
        class_id=[14],
    )
    # the five classes below are used in a traffic_cone detection model
    # required by SD project
    # unlike traffic_cone, only cone class in this item, no subclasses
    cone_detection = CloudModelTask(
        task_name="cone_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["cone"],
        class_id=[1],
    )
    traffic_bollard_detection = CloudModelTask(
        task_name="traffic_bollard_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["traffic_bollard"],
        class_id=[2],
    )
    isolation_bollard_detection = CloudModelTask(
        task_name="isolation_bollard_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["isolation_bollard"],
        class_id=[3],
    )
    crash_barrel_detection = CloudModelTask(
        task_name="crash_barrel_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["crash_barrel"],
        class_id=[4],
    )
    aframe_sign_detection = CloudModelTask(
        task_name="aframe_sign_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["aframe_sign"],
        class_id=[5],
    )
    adb_light_detection = CloudModelTask(
        task_name="adb_light_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["adb_light"],
        subclass_name=[
            "head_light",
            "tail_light",
            "street_light",
            "reflection_point",
        ],
        class_id=[11, 12, 13, 14],
    )
    vehicle_side_detection = CloudModelTask(
        task_name="vehicle_side_detection",
        task_type=TaskType.detection,
        model_type=ModelType.image_model,
        class_name=["vehicle_side"],
        class_id=[11],
    )
    # ------------------------------------------
    # segmentations
    # ------------------------------------------
    lane_instanceseg = CloudModelTask(
        task_name="lane_instanceseg",
        task_type=TaskType.instanceseg,
        model_type=ModelType.image_model,
        class_name=["lane"],
        class_id=[-1],
    )
    semantic_parsing_40cls_segmentation = CloudModelTask(
        task_name="semantic_parsing_40cls_segmentation",
        task_type=TaskType.segmentation,
        model_type=ModelType.image_model,
        class_name=["semantic_parsing_40cls"],
        class_id=[-1],
    )
    semantic_parsing_33cls_segmentation = CloudModelTask(
        task_name="semantic_parsing_33cls_segmentation",
        task_type=TaskType.segmentation,
        model_type=ModelType.image_model,
        class_name=["semantic_parsing_33cls"],
        class_id=[-1],
    )
    # ------------------------------------------
    # classification
    # ------------------------------------------
    vehicle_attribute_classification = CloudModelTask(
        task_name="vehicle_attribute_classification",
        task_type=TaskType.classification,
        model_type=ModelType.roi_model,
        class_name=[
            "vehicle_attribute",
        ],
        class_id=[-1],
    )
    vehicle_rear_attribute_classification = CloudModelTask(
        task_name="vehicle_rear_attribute_classification",
        task_type=TaskType.classification,
        model_type=ModelType.roi_model,
        class_name=[
            "vehicle_rear_attribute",
        ],
        class_id=[-1],
    )
    person_attribute_classification = CloudModelTask(
        task_name="person_attribute_classification",
        task_type=TaskType.classification,
        model_type=ModelType.roi_model,
        class_name=[
            "person_attribute",
        ],
        class_id=[-1],
    )
    traffic_light_primary_attribute_classification = CloudModelTask(
        task_name="traffic_light_primary_attribute_classification",
        task_type=TaskType.classification,
        model_type=ModelType.roi_model,
        class_name=[
            "traffic_light_primary_attribute",
        ],
        class_id=[-1],
    )
    traffic_light_secondary_attribute_classification = CloudModelTask(
        task_name="traffic_light_secondary_attribute_classification",
        task_type=TaskType.classification,
        model_type=ModelType.roi_model,
        class_name=[
            "traffic_light_secondary_attribute",
        ],
        class_id=[-1],
    )
    traffic_cone_attribute_classification = CloudModelTask(
        task_name="traffic_cone_attribute_classification",
        task_type=TaskType.classification,
        model_type=ModelType.roi_model,
        class_name=[
            "traffic_cone_attribute",
        ],
        class_id=[-1],
    )
    road_arrow_attribute_classification = CloudModelTask(
        task_name="road_arrow_attribute_classification",
        task_type=TaskType.classification,
        model_type=ModelType.roi_model,
        class_name=[
            "road_arrow_attribute",
        ],
        class_id=[-1],
    )
    traffic_sign_medium_attribute_classification = CloudModelTask(
        task_name="traffic_sign_medium_attribute_classification",
        task_type=TaskType.classification,
        model_type=ModelType.roi_model,
        class_name=[
            "traffic_sign_medium_attribute",
        ],
        class_id=[-1],
    )
    cyclist_wheel_attribute_classification = CloudModelTask(
        task_name="cyclist_wheel_attribute_classification",
        task_type=TaskType.classification,
        model_type=ModelType.roi_model,
        class_name=[
            "cyclist_wheel_attribute",
        ],
        class_id=[-1],
    )
    vehicle_light_detection_attribute_classification = CloudModelTask(
        task_name="vehicle_light_detection_attribute_classification",
        task_type=TaskType.classification,
        model_type=ModelType.roi_model,
        class_name=[
            "vehicle_light_detection_attribute",
        ],
        class_id=[-1],
    )
    wheel_attribute_classification = CloudModelTask(
        task_name="wheel_attribute_classification",
        task_type=TaskType.classification,
        model_type=ModelType.roi_model,
        class_name=[
            "wheel_attribute",
        ],
        class_id=[-1],
    )
    face_attribute_classification = CloudModelTask(
        task_name="face_attribute_classification",
        task_type=TaskType.classification,
        model_type=ModelType.roi_model,
        class_name=[
            "face_attribute",
        ],
        class_id=[-1],
    )
    person_head_attribute_classification = CloudModelTask(
        task_name="person_head_attribute_classification",
        task_type=TaskType.classification,
        model_type=ModelType.roi_model,
        class_name=[
            "person_head_attribute",
        ],
        class_id=[-1],
    )
    vehicle_light_recognition_classification = CloudModelTask(
        task_name="vehicle_light_recognition_classification",
        task_type=TaskType.classification,
        model_type=ModelType.roi_model,
        class_name=[
            "vehicle_light_recognition",
        ],
        class_id=[-1],
    )
    traffic_sign_recognition_classification = CloudModelTask(
        task_name="traffic_sign_recognition_classification",
        task_type=TaskType.classification,
        model_type=ModelType.roi_model,
        class_name=[
            "traffic_sign_recognition",
        ],
        class_id=[-1],
    )
    vehicle_negative_recognition_classification = CloudModelTask(
        task_name="vehicle_negative_recognition_classification",
        task_type=TaskType.classification,
        model_type=ModelType.roi_model,
        class_name=[
            "vehicle_negative_recognition",
        ],
        class_id=[-1],
    )
    person_orientation_recognition_classification = CloudModelTask(
        task_name="person_orientation_recognition_classification",
        task_type=TaskType.classification,
        model_type=ModelType.roi_model,
        class_name=[
            "person_orientation_recognition",
        ],
        class_id=[-1],
    )

    @classmethod
    def class_name2tasks(cls):
        class_name2tasks = defaultdict(set)
        for task_i in vars(cls).values():
            if isinstance(task_i, CloudModelTask):
                for class_name_i in task_i.class_name:
                    class_name2tasks[class_name_i].add(task_i)
        class_name2tasks = {k: list(v) for k, v in class_name2tasks.items()}
        return class_name2tasks

    @classmethod
    def all_tasks(cls):
        return [i for i in vars(cls).values() if isinstance(i, CloudModelTask)]
