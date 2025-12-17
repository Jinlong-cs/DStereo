from typing import Dict, List, Optional, Union

import cv2
import numpy as np

# 75 RGB:
_COLORS = np.array(
    [
        [0.0, 0.447, 0.741],  # noqa
        [0.85, 0.325, 0.098],  # noqa
        [0.929, 0.694, 0.125],  # noqa
        [0.494, 0.184, 0.556],  # noqa
        [0.466, 0.674, 0.188],  # noqa
        [0.301, 0.745, 0.933],  # noqa
        [0.635, 0.078, 0.184],  # noqa
        [0.3, 0.3, 0.3],  # noqa
        [0.6, 0.6, 0.6],  # noqa
        [1.0, 0.0, 0.0],  # noqa
        [1.0, 0.5, 0.0],  # noqa
        [0.749, 0.749, 0.0],  # noqa
        [0.0, 1.0, 0.0],  # noqa
        [0.0, 0.0, 1.0],  # noqa
        [0.667, 0.0, 1.0],  # noqa
        [0.333, 0.333, 0.0],  # noqa
        [0.333, 0.667, 0.0],  # noqa
        [0.333, 1.0, 0.0],  # noqa
        [0.667, 0.333, 0.0],  # noqa
        [0.667, 0.667, 0.0],  # noqa
        [0.667, 1.0, 0.0],  # noqa
        [1.0, 0.333, 0.0],  # noqa
        [1.0, 0.667, 0.0],  # noqa
        [1.0, 1.0, 0.0],  # noqa
        [0.0, 0.333, 0.5],  # noqa
        [0.0, 0.667, 0.5],  # noqa
        [0.0, 1.0, 0.5],  # noqa
        [0.333, 0.0, 0.5],  # noqa
        [0.333, 0.333, 0.5],  # noqa
        [0.333, 0.667, 0.5],  # noqa
        [0.333, 1.0, 0.5],  # noqa
        [0.667, 0.0, 0.5],  # noqa
        [0.667, 0.333, 0.5],  # noqa
        [0.667, 0.667, 0.5],  # noqa
        [0.667, 1.0, 0.5],  # noqa
        [1.0, 0.0, 0.5],  # noqa
        [1.0, 0.333, 0.5],  # noqa
        [1.0, 0.667, 0.5],  # noqa
        [1.0, 1.0, 0.5],  # noqa
        [0.0, 0.333, 1.0],  # noqa
        [0.0, 0.667, 1.0],  # noqa
        [0.0, 1.0, 1.0],  # noqa
        [0.333, 0.0, 1.0],  # noqa
        [0.333, 0.333, 1.0],  # noqa
        [0.333, 0.667, 1.0],  # noqa
        [0.333, 1.0, 1.0],  # noqa
        [0.667, 0.0, 1.0],  # noqa
        [0.667, 0.333, 1.0],  # noqa
        [0.667, 0.667, 1.0],  # noqa
        [0.667, 1.0, 1.0],  # noqa
        [1.0, 0.0, 1.0],  # noqa
        [1.0, 0.333, 1.0],  # noqa
        [1.0, 0.667, 1.0],  # noqa
        [0.333, 0.0, 0.0],  # noqa
        [0.5, 0.0, 0.0],  # noqa
        [0.667, 0.0, 0.0],  # noqa
        [0.833, 0.0, 0.0],  # noqa
        [1.0, 0.0, 0.0],  # noqa
        [0.0, 0.167, 0.0],  # noqa
        [0.0, 0.333, 0.0],  # noqa
        [0.0, 0.5, 0.0],  # noqa
        [0.0, 0.667, 0.0],  # noqa
        [0.0, 0.833, 0.0],  # noqa
        [0.0, 1.0, 0.0],  # noqa
        [0.0, 0.0, 0.167],  # noqa
        [0.0, 0.0, 0.333],  # noqa
        [0.0, 0.0, 0.5],  # noqa
        [0.0, 0.0, 0.667],  # noqa
        [0.0, 0.0, 0.833],  # noqa
        [0.0, 0.0, 1.0],  # noqa
        [0.0, 0.0, 0.0],  # noqa
        [0.143, 0.143, 0.143],  # noqa
        [0.857, 0.857, 0.857],  # noqa
        [1.0, 1.0, 1.0],
    ],  # noqa
    dtype=np.float32,
)


def colormap(rgb: Optional[bool] = False, maximum: Optional[int] = 255):
    assert maximum in [255, 1], maximum
    c = _COLORS * maximum
    if not rgb:
        c = c[:, ::-1]
    return c


def random_color(rgb: Optional[bool] = False, maximum: Optional[int] = 255):
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


def get_color_board(board_path: Optional[str] = "color_board.png"):
    size = 100
    hh, ww = 10, 10
    canvas = np.random.rand(hh * size, ww * size, 3).astype("float32")
    for h in range(hh):
        for w in range(ww):
            idx = h * ww + w
            if idx >= len(_COLORS):
                break
            canvas[h * size : (h + 1) * size, w * size : (w + 1) * size] = [
                int(i * 255) for i in _COLORS[idx][::-1]
            ]
    cv2.imwrite(board_path, canvas)
    return board_path


class ClassMetaBase:
    # define colors for all bbox related tasks(4PE/2PE).
    # Note Since We Need to support AdasInference and Hpflow,
    # some class names maybe needed in different names.

    def __init__(self, ltc_project_name: Optional[str] = None):
        # thing and stuff colors
        self.ltc_project_name = ltc_project_name
        # classification topics
        self.image_cls_colors = dict()
        # detection
        self.thing_colors = dict()
        self.thing_classes = list(self.thing_colors.keys())
        self.real3d_instance = dict()
        # parsing_colormap
        self.parsing_colormap = None
        self.lane_parsing_colormap = None
        # kps
        self.keypoint_task_to_names = dict()
        self.keypoint_names = dict()
        self.keypoint_connection_rules = dict()
        # vis issue
        self.task_name_map = dict()
        self.class_name2position = dict()

    def update_thing_class_and_color(
        self, class_names: List[str], color_maps: Optional[Dict]
    ):
        """Update classes and color mappings by user defined color_maps.

        Args:
            class_names (str): List of class_names need to be visualized.
            color_maps (Dict): Dict, {class_name: color}
                the second one is idx of color_map (see _COLORS).
                Could only define parts of colors, other classes will be
                assigned color randomly. For each target class, could also
                define colors for ignore region and hard_instance by defining
                "{classname}_ignore" and "{classname}_hard".
        """

        new_thing_color_mappings = {}
        new_class_name2position = {}
        new_thing_classes = []

        def set_class_for_one_classname(classname: str):
            if classname in color_maps:
                color = _COLORS[int(color_maps[classname])]
            else:
                color = random_color(rgb=True, maximum=1)
            new_thing_color_mappings[classname] = color
            new_class_name2position[
                classname
            ] = "left"  # TODO: allow defining position if need
            new_thing_classes.append(classname)

        for classname in class_names:
            # color for normal instances
            set_class_for_one_classname(classname)
            set_class_for_one_classname(classname + "_hard")
            set_class_for_one_classname(classname + "_ignore")

        self.thing_colors = new_thing_color_mappings
        self.thing_classes = new_thing_classes
        self.class_name2position = new_class_name2position

    def update_parsing_color_map(
        self, color_map: Union[Dict[str, List[int]], Dict[str, int]]
    ):
        self.parsing_colormap = color_map

    def update_lane_parsing_color_map(
        self, color_map: Union[Dict[str, List[int]], Dict[str, int]]
    ):
        self.lane_parsing_colormap = color_map

    def update_keypoint_labels(
        self,
        keypoint_task_to_names,
        keypoint_names,
        keypoint_connection_rules,
    ):
        self.keypoint_task_to_names = keypoint_task_to_names
        self.keypoint_names = keypoint_names
        self.keypoint_connection_rules = keypoint_connection_rules
