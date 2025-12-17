import os
import warnings
from enum import Enum, unique
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

__all__ = [
    "check_obj",
    "drawcontour",
    "imread",
    "imwrite",
    "draw_mask",
    "check_parsing_ignore",
    "verify_image",
    "get_image_shape",
]

numeric_types = (float, int, np.generic)


@unique
class ImageFailCtsType(Enum):
    FULL_IMAGE = 1
    PARTIAL_POSSIBLE = 2
    PARTIAL_CERTAIN = 3
    IGNORE = 4


def image_fail_parsing_anno_to_contours_fn(ct, width, height, labels):
    """
    Transfrom the annotation data to contours for image fail parsing.

    Parameters
    ----------
    ct: dict
        the contour data.
    width: int
        the image width.
    height: int
        the image height.
    labels: dict
        the label map.
    """
    pts = ct["data"]
    if None in pts:
        return None
    ct_pts = []
    for pt in pts:
        ct_pts.append(
            (
                max(min(int(float(pt[0])), width - 1), 0),
                max(min(int(float(pt[1])), height - 1), 0),
            )
        )
    ct_pts = np.array(ct_pts)

    value = 0
    anno_type = ct["attrs"]["type"]
    anno_degree = ct["attrs"]["degree"]
    if anno_type == "normal":
        anno_label_name = anno_type
    else:
        anno_label_name = f"{anno_degree}_{anno_type}"
    attrs = ct["attrs"]
    if attrs["ignore"] == "yes" or attrs["confidence"] == "possible":
        value = 255
    elif anno_label_name in labels:
        value = labels[anno_label_name]
    else:
        value = 255
    # check night_light
    # anno_night_light = 'no_light'
    # if 'night_light' in ct['attrs']:
    #     anno_night_light = ct['attrs']['night_light']
    #     if anno_type == 'blur' and anno_night_light == 'dark_night_light':
    #         value = labels['normal']
    return ct_pts, value


def anno_to_contours_fn_with_label_mapping(
    data: Dict, width: int, height: int, label_map_dict: Dict
):
    """
    Transfrom the annotation data to contours.

    Args:
        data:
            the annotation label.
        width:
            the image width.
        height:
            the image height.
        label_map_dict:
            the label map.
    """
    pts = data["data"]
    if None in pts:
        return None
    ct_pts = []
    for pt in pts:
        ct_pts.append(
            (
                max(min(int(float(pt[0])), width - 1), 0),
                max(min(int(float(pt[1])), height - 1), 0),
            )
        )
    ct_pts = np.array(ct_pts)
    match_flag, value = is_label_match(data["attrs"], label_map_dict)
    assert match_flag, "data attrs dont match with label mapping conditions"
    return ct_pts, value


def default_anno_to_contours_fn(
    data: Dict, width: int, height: int, labels: Dict
):
    """
    Transfrom the annotation data to contours.

    Parameters
    ----------
    data:
        the annotation label.
    width:
        the image width.
    height:
        the image height.
    labels:
        the label map.
    """
    pts = data["data"]
    if None in pts:
        return None
    ct_pts = []
    for pt in pts:
        ct_pts.append(
            (
                max(min(int(float(pt[0])), width - 1), 0),
                max(min(int(float(pt[1])), height - 1), 0),
            )
        )
    ct_pts = np.array(ct_pts)

    value = 0
    value = labels[data["attrs"]["type"]]
    if "ignore" in data["attrs"] and data["attrs"]["ignore"] == "yes":
        value = 255
    elif (
        "double_line" in data["attrs"]
        and data["attrs"]["double_line"] == "yes"
    ):  # noqa
        value = labels["double_line"]
    assert value != -1, "{}, {}".format(
        data["attrs"]["type"], data["attrs"]["ignore"]
    )  # noqa
    return ct_pts, value


def is_label_match(data_attrs: Dict, label_map_dict: Dict):
    """
    Get the mapping according to conditions of label map dict.

    Parameters
    ----------
    data_attrs:
        the labeled information.
    label_map_dict:
        the label mao dict including match conditions and label value.
    """
    for label_name in label_map_dict.keys():
        match_conditions_list = label_map_dict[label_name]["match_conditions"]
        value = label_map_dict[label_name]["id"]
        for match_conditions in match_conditions_list:
            for match_key in match_conditions.keys():
                match_flag = False
                if data_attrs[match_key] in match_conditions[match_key]:
                    match_flag = True
                else:
                    match_flag = False
                    break
            if match_flag is True:
                return match_flag, value
    if match_flag is False:
        return match_flag, None


def get_map(src_label: Dict, dst_label: Dict):
    """
    Get the mapping from src_label to dst_label.

    Args:
        src_label:
            the source label.
        dst_label:
            the destination label.
    """
    assert len(src_label.keys()) == len(dst_label.keys()), "{},{}".format(
        len(src_label.keys()), len(dst_label.keys())
    )
    anno_map = 256 * np.ones(256)
    for i in src_label.keys():
        if anno_map[src_label[i]] < 256:
            assert anno_map[src_label[i]] == dst_label[i], (
                "Please check the value of key {} (value: {})."
                "The value is conflict with other key(s). {} vs {}".format(
                    i, src_label[i], anno_map[src_label[i]], dst_label[i]
                )
            )
        anno_map[src_label[i]] = dst_label[i]
    return anno_map


def draw_colorbar(colors: List[int], clsnames: List[str], outpath: str):
    """Draw color bar for segmentation task.

    Args:
        colors:
            RGB color of each segmentation class.
        clsnames:
            segmentation class names.
        outpath:
            Color bar output path.
    """

    def putText(im, str, position, scale):
        img_PIL = Image.fromarray(im[:, :, ::-1])
        ImageDraw.Draw(img_PIL).text(position, str, fill=(255, 255, 255))
        return np.asarray(img_PIL)[:, :, ::-1]

    num_cls = len(clsnames)

    max_c = 0
    for name in clsnames:
        max_c = max(max_c, len(name))

    length = int(720 / int(num_cls))
    bar = np.zeros((720, int(0.4 * length) * (max_c + 2), 3))
    for i in range(int(num_cls)):
        bar[i * length : (i + 1) * length, :, 0] = colors[i, :, 2]
        bar[i * length : (i + 1) * length, :, 1] = colors[i, :, 1]
        bar[i * length : (i + 1) * length, :, 2] = colors[i, :, 0]
    bar[int(num_cls) * length :, :, 0] = colors[int(num_cls) - 1, :, 2]
    bar[int(num_cls) * length :, :, 1] = colors[int(num_cls) - 1, :, 1]
    bar[int(num_cls) * length :, :, 2] = colors[int(num_cls) - 1, :, 0]

    bar_show = bar.astype("uint8")
    for i in range(int(num_cls)):
        str = "%02d %s" % (i, clsnames[i])
        bar_show = putText(
            bar_show,
            str,
            (int(0.2 * length), int(i * length)),
            int(0.6 * length),
        )

    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    cv2.imwrite(
        os.path.join(outpath, "colorbar_{}.png".format(num_cls)),
        bar_show.astype("uint8"),
    )


def _default_check_render_image(image_path: str):
    """Check given the path of an image is a render image.

    If is a render image, return the path of source image.

    Args:
        image_path:
            the absolute path of an image.
    """
    image_path = os.path.abspath(image_path)
    assert (
        image_path.endswith("jpg")
        or image_path.endswith("jpeg")
        or image_path.endswith("png")
    ), ("invalid ext for image %s, only allow jpg, jpeg, png" % image_path)

    if "_render" in image_path:
        assert image_path.endswith("_render.jpg")
        if os.path.isfile(image_path.replace("_render.jpg", ".jpeg")):
            image_path = image_path.replace("_render.jpg", ".jpeg")
        elif os.path.isfile(image_path.replace("_render.jpg", ".jpg")):
            image_path = image_path.replace("_render.jpg", ".jpg")
        else:
            raise RuntimeError(
                "error! cannot find raw image for rendered image %s!"
                % (image_path)
            )  # noqa
    return image_path


def _is_valid_jpg(jpg_file):
    with open(jpg_file, "rb") as f:
        f.seek(-10, 2)
        buf = f.read()
        return b"\xff\xd9" in buf


def check_image_completeness(image_path):
    """Check whether an image is broken or not."""
    if image_path.endswith("jpeg") or image_path.endswith("jpg"):
        return _is_valid_jpg(image_path)

    elif image_path.endswith("png"):
        return True
    else:
        raise NotImplementedError(
            "Check image completeness temporary support only .jpg and .png"
        )


# obj_filter
def get_bbox_x1(obj: Dict):
    """Get bounding box Left coordinate (x1).

    Args:
        obj:
            Bouding box object, {"data": [x1, y1, x2, y2]}
    """
    x1 = float(obj["data"][0])
    return x1


def get_bbox_y1(obj: Dict):
    """Get bounding box top coordinate (y1).

    Args:
        obj:
            Bouding box object, {"data": [x1, y1, x2, y2]}
    """
    y1 = float(obj["data"][1])
    return y1


def get_bbox_x2(obj: Dict):
    """Get bounding box right coordinate (x2).

    Args:
        obj:
            Bouding box object, {"data": [x1, y1, x2, y2]}
    """
    x2 = float(obj["data"][2])
    return x2


def get_bbox_y2(obj: Dict):
    """Get bounding box bottom coordinate (y2).

    Args:
        obj: dict
            Bouding box object, {"data": [x1, y1, x2, y2]}
    """
    y2 = float(obj["data"][3])
    return y2


def get_bbox_height(obj: Dict):
    """Get bounding box height (y2 - y1).

    Args:
        obj:
            Bouding box object, {"data": [x1, y1, x2, y2]}
    """
    x1, y1, x2, y2 = map(float, obj["data"])
    height = y2 - y1
    return height


def get_bbox_width(obj: Dict):
    """Get bounding box width (x2 - x1).

    Args:
        obj:
            Bouding box object, {"data": [x1, y1, x2, y2]}
    """
    x1, y1, x2, y2 = map(float, obj["data"])
    width = x2 - x1
    return width


def get_bbox_shortside(obj: Dict):
    """Get bounding box shortside length (min(width, height)).

    Args:
        obj:
            Bouding box object, {"data": [x1, y1, x2, y2]}
    """
    height = get_bbox_height(obj)
    width = get_bbox_width(obj)
    return min(height, width)


def get_bbox_aspect_ratio(obj: Dict):
    """Get bounding box aspect ratio (height / width).

    Args:
        obj:
            Bouding box object, {"data": [x1, y1, x2, y2]}
    """
    return get_bbox_height(obj) / float(get_bbox_width(obj))


def get_value(obj, field: str):
    """Get object attribute value.

    Args:
        field:
    """
    value = obj
    for key in field.split("."):
        value = value[key]
    return value


def _contains(lhs_dict: Dict, rhs_dict: Dict):
    """Check if the left dict contains the right dict.

    Args:
        lhs_dict:
        rhs_dict:
    """
    if isinstance(lhs_dict, dict) and isinstance(rhs_dict, dict):
        for key in rhs_dict:
            if key not in lhs_dict:
                return False
            if not _contains(lhs_dict[key], rhs_dict[key]):
                return False
    else:
        return lhs_dict == rhs_dict
    return True


def _has_path(lhs_dict: Dict, path: Dict):
    """Check if the dict has a specific path.

    Args:
        lhs_dict:
        rhs_dict:
    """
    if isinstance(path, str):
        if isinstance(lhs_dict, dict):
            pos = path.find(".")
            if pos > 0:
                key = path[:pos]
                subpath = path[pos + 1 :]
                return _has_path(lhs_dict.get(key), subpath)
            else:
                key = path
                return key in lhs_dict
        elif isinstance(lhs_dict, str):
            if path.find(".") > 0:
                return False
            else:
                return lhs_dict == path
    return False


def _range(obj: Dict, range_condiction: Dict):
    """Check whether the object meets the range condition.

    Args:
        obj:
        range_condiction:
    """
    field = range_condiction["field"]
    if field.startswith("$"):
        if field == "$BBOX_HEIGHT":
            value = get_bbox_height(obj)
        elif field == "$BBOX_WIDTH":
            value = get_bbox_width(obj)
        elif field == "$BBOX_ASPECT_RATIO":
            value = get_bbox_aspect_ratio(obj)
        elif field == "$BBOX_SHORTSIDE":
            value = get_bbox_shortside(obj)
        elif field == "$BBOX_X1":
            value = get_bbox_x1(obj)
        elif field == "$BBOX_Y1":
            value = get_bbox_y1(obj)
        elif field == "$BBOX_X2":
            value = get_bbox_x2(obj)
        elif field == "$BBOX_Y2":
            value = get_bbox_y2(obj)
        else:
            raise Exception("Invalid range condiction %s" % field)
    else:
        value = get_value(obj, field)

    if "lt" in range_condiction and not value < range_condiction["lt"]:
        return False
    elif "le" in range_condiction and not value <= range_condiction["le"]:
        return False
    elif "gt" in range_condiction and not value > range_condiction["gt"]:
        return False
    elif "ge" in range_condiction and not value >= range_condiction["ge"]:
        return False
    else:
        return True


def _and(obj, condictions):
    """Check whether the object meets all conditions.

    Args:
        obj:
        condictions: of dict
            List of condition
    """
    for condiction in condictions:
        if not check_obj(obj, condiction):
            return False
    return True


def _or(obj, condictions):
    """Check whether the object meets any conditions.

    Args:
        obj:
        condictions: of dict
            List of condition
    """
    for condiction in condictions:
        if check_obj(obj, condiction):
            return True
    return False


def _not(obj: dict, condiction: dict):
    """Check whether the object does not meets condition.

    Args:
        obj:
        condiction:
    """
    return not check_obj(obj, condiction)


_ops = {
    "or": _or,
    "and": _and,
    "not": _not,
    "contains": _contains,
    "range": _range,
    "has_path": _has_path,
}


def check_obj(obj: dict, condiction: dict):
    """Check whether the object meets the condition.

    Args:
        obj: dict
        condiction: dict
    """
    if condiction is None:
        return False
    for op_name in _ops:
        if op_name in condiction:
            return _ops[op_name](obj, condiction[op_name])
    raise Exception("Invalid condiction keys: %s" % condiction.keys())


def drawcontour(contour: np.ndarray, value: np.ndarray, J: object):
    """Draw contour.

    Args:
        contour:
            The input image.
        value:
            The segmentation label map.
        J:
            label colors.
    """
    if value > 255:
        return J
    M = np.zeros(J.shape, dtype=np.uint8)
    cv2.drawContours(M, [contour], -1, (1), -1)
    J[(M == 1)] = value
    return J


def imread(filename: str):
    """Read image.

    Args:
        filename:
            The image path.
    """
    try:
        im = cv2.imread(filename, -1)
    except Exception:
        im = cv2.imdecode(
            np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
    return im


def imwrite(filename: str, im: np.ndarray):
    """Write image.

    Args:
        filename:
            The image path.
        im:
            The image.
    """
    try:
        im = cv2.imwrite(filename, im)
    except Exception:
        im = cv2.imencode(os.path.splitext(filename)[1], im)[1].tofile(
            filename
        )


def fuse_mask(
    img: np.ndarray, label: np.ndarray, colors: object
) -> np.ndarray:
    """Fuse label mask.

    Args:
        img:
            The input image.
        label:
            The segmentation label map.
        colors:
            label colors.
    """
    bgr = cv2.cvtColor(label.astype("uint8"), cv2.COLOR_GRAY2BGR)
    show = np.array(cv2.LUT(bgr, colors))
    fusion = (
        show.astype("float32") * 0.5 + np.array(img).astype("float32") * 0.5
    ).astype("uint8")
    return fusion


def draw_mask(
    img: np.ndarray, label: np.ndarray, colors: object
) -> np.ndarray:
    """Draw label mask.

    Args:
        img:
            The input image.
        label:
            The segmentation label map.
        colors:
            label colors.
    """
    fusion = fuse_mask(img, label, colors)
    return np.vstack((img, fusion)).astype("uint8")


def check_parsing_ignore(label_path: str, ignore_thres: int = 0.8) -> bool:
    """Verify annotation of a label file for segmentation.

       If ratio of ignore region > ignore_thres, skip it.

    Args:
        label_path:
            label path.

    Returns:
        True for valid label, otherwise False.
    """
    if not os.path.exists(label_path):
        warnings.warn("cannot find label %s, ignoring..." % label_path)
        return False
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    ignore_size = np.sum(label == 255)
    if ignore_size > label.shape[0] * label.shape[1] * ignore_thres:
        warnings.warn(
            "Too many pixels are ignored in label file %s, ignoring..."
            % label_path
        )
        return False
    return True


def _get_box_10_points(bbox):
    x1, y1, x2, y2 = bbox
    height = y2 - y1
    width = x2 - x1
    cx = x1 + width / 2.0
    cy = y1 + height / 2.0
    points_data = [
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
        [cx, cy],
        [cx + (width + height) / 2.0, cy],
        [cx + (width * height) ** 0.5, cy],
        [cx + max(width, height), cy],
        [cx + max(2 * width, height), cy],
        [cx + max(width, 2 * height), cy],
    ]

    return points_data


def verify_image(
    image_path: str, check_image_invalid: Optional[bool] = True
) -> bool:
    """Verify validity of an image.

    Args:
        image_path:
            Image path.
        check_image_invalid:
            Whether to check the image invalid.

    Returns:
        True for valid image, otherwise False.
    """
    image_path = os.path.expanduser(image_path)
    if not os.path.exists(image_path):
        warnings.warn("cannot find image %s, ignoring..." % image_path)
        return False
    if check_image_invalid:
        img = cv2.imread(image_path)
        if img is None:
            warnings.warn("invalid image %s, ignoring..." % image_path)
            return False
    return True


def get_image_shape(image):
    img_h, img_w = image.shape[:2]
    img_c = 1 if len(image.shape) == 2 else image.shape[2]
    return img_h, img_w, img_c
