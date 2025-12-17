from projects.cloudmodel.tools.visualizer.class_metas.utils import (
    _COLORS,
    ClassMetaBase,
)

__all__ = ["ClassMetaCloudModel"]


class ClassMetaCloudModel(ClassMetaBase):
    # define colors for all bbox related tasks(4PE/2PE).
    # Note Since We Need to support AdasInference and Hpflow,
    # some class names maybe needed in different names.
    _thing_colors = dict(
        # dynamic
        vehicle=_COLORS[49],
        vehicle_wheel=_COLORS[50],
        vehicle_light=_COLORS[51],
        vehicle_plate=_COLORS[52],
        vehicle_side=_COLORS[53],
        cyclist=_COLORS[3],
        cyclist_wheel=_COLORS[4],
        rear=_COLORS[30],
        person=_COLORS[9],
        person_head=_COLORS[10],
        person_face=_COLORS[11],
        cycle=_COLORS[12],
        # adb light
        head_light=_COLORS[39],
        tail_light=_COLORS[40],
        street_light=_COLORS[41],
        reflection_point=_COLORS[42],
        # static
        traffic_sign=_COLORS[23],
        traffic_cone=_COLORS[20],
        cone=_COLORS[20],
        traffic_light=_COLORS[21],
        traffic_light_len=_COLORS[22],
        road_arrow=_COLORS[23],
        traffic_bollard=_COLORS[24],
        isolation_bollard=_COLORS[25],
        crash_barrel=_COLORS[26],
        aframe_sign=_COLORS[27],
        # add a unknown
        unknown=_COLORS[70],
    )

    _class_name2position = {
        # dynamic
        "vehicle": "right",
        "rear": "left",
        "cyclist": "right",
        "person": "left",
        "head_light": "left",
        "tail_light": "left",
        "street_light": "left",
        "reflection_point": "left",
        # static
        "traffic_sign": "left",
        "traffic_cone": "left",
        "traffic_light": "left",
        "road_arrow": "left",
        # static_sd
        "traffic_bollard": "left",
        "isolation_bollard": "left",
        "crash_barrel": "left",
        "aframe_sign": "left",
        "cone": "left",
        # add an unknown
        "unknown": "left",
    }

    _attribute_topic_map = {
        "lane_type": "type",
        "lane_double_line": "double_line",
        "lane_color": "color",
        "lane_occlusion": "occlusion",
        "scene_classification": "scene",
        "weather_classification": "weather",
        "illumination_classification": "illumination",
        "time_classification": "time",
        # traffic sign
        "cn_sub_type": "CnSubType",
    }
    _classification_colors = dict(
        # image level classification class topics
        work_condition=_COLORS[2],
    )
    _keypoint_task_to_names = dict(
        vehicle_side_detection=["vehicle_side"],
    )
    _keypoint_names = dict(
        # wheel keypoints
        vehicle_side=dict(
            name=["lt", "rt", "rb", "lb"],
            color=[_COLORS[1], _COLORS[2], _COLORS[3], _COLORS[4]],
        ),
    )
    _keypoint_connection_rules = dict(
        vehicle_side=[
            # line from point1 --> point2
            ["lt", "rt", _COLORS[53]],
            ["rt", "rb", _COLORS[53]],
            ["rb", "lb", _COLORS[53]],
            ["lb", "lt", _COLORS[53]],
        ],
    )

    def __init__(self):
        # classification topics
        self.image_cls_colors = self._classification_colors
        # detection
        self.thing_colors = self._thing_colors
        self.thing_classes = list(self._thing_colors.keys())
        # parsing
        self.parsing_colormap = self.get_parsing_classmaps()
        self.lane_parsing_colormap = self.get_lane_parsing_classmaps()
        # kps
        self.keypoint_task_to_names = self._keypoint_task_to_names
        self.keypoint_names = self._keypoint_names
        self.keypoint_connection_rules = self._keypoint_connection_rules
        # vis issue
        self.task_name_map = self._attribute_topic_map
        self.class_name2position = self._class_name2position

    def get_parsing_classmaps(self):
        from hat.core.proj_spec.parsing import get_default_parsing_labels

        color_maps = {
            "semantic_parsing_40cls_segmentation": {
                i["class_name"][0]: i["color_map"]
                for i in get_default_parsing_labels("gl_40")
            },
        }
        return color_maps

    def get_lane_parsing_classmaps(self):
        lane_desc_labels_13cls = [
            {"class_name": "background", "color_map": [0, 0, 0]},
            {"class_name": "word", "color_map": [119, 11, 32]},
            {"class_name": "imprint", "color_map": [0, 0, 142]},
            {"class_name": "deceleration_lane", "color_map": [0, 0, 230]},
            {"class_name": "light_shadow", "color_map": [106, 0, 228]},
            {"class_name": "shadow", "color_map": [0, 60, 100]},
            {"class_name": "rain_trace", "color_map": [0, 80, 100]},
            {"class_name": "one2two", "color_map": [0, 0, 70]},
            {"class_name": "old_lane", "color_map": [0, 0, 192]},
            {"class_name": "reflection", "color_map": [250, 170, 30]},
            {"class_name": "seam", "color_map": [100, 170, 30]},
            {"class_name": "light", "color_map": [220, 220, 0]},
            {"class_name": "grid", "color_map": [175, 116, 175]},
        ]

        lane_desc_labels_7cls = [
            {"class_name": "background", "color_map": [0, 0, 0]},
            {"class_name": "lane", "color_map": [0, 0, 255]},
            {"class_name": "curb", "color_map": [0, 255, 0]},
            {"class_name": "double_line", "color_map": [255, 255, 0]},
            {"class_name": "wide_dashed", "color_map": [190, 153, 153]},
            {"class_name": "wide_solid", "color_map": [153, 51, 204]},
            {"class_name": "deceleration_lane", "color_map": [0, 255, 255]},
        ]
        color_maps = {
            "lane_7cls_segmentation": {
                i["class_name"]: i["color_map"] for i in lane_desc_labels_7cls
            },
            "badcase_parsing_13cls_segmentation": {
                i["class_name"]: i["color_map"] for i in lane_desc_labels_13cls
            },
        }
        return color_maps
