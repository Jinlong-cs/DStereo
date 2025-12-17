import numpy as np

__all__ = ["norm_bbox"]


def norm_bbox(bbox, norm_method, norm_ratio, img_shape):
    box_w = bbox[2] - bbox[0]
    box_h = bbox[3] - bbox[1]
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    w_new = box_w
    h_new = box_h

    if norm_method == "width_length":
        w_new = box_w * norm_ratio
        h_new = box_h + w_new - box_w
        assert h_new > 0
    elif norm_method == "height_length":
        h_new = box_h * norm_ratio
        w_new = box_w + h_new - box_h
        assert w_new > 0
    elif norm_method == "longside_length":
        if box_w > box_h:
            w_new = box_w * norm_ratio
            h_new = box_h + w_new - box_w
            assert h_new > 0
        else:
            h_new = box_h * norm_ratio
            w_new = box_w + h_new - box_h
            assert w_new > 0
    elif norm_method in ["width_ratio", "height_ratio", "longside_ratio"]:
        h_new = box_h * norm_ratio
        w_new = box_w * norm_ratio
    elif norm_method == "longside_square":
        if box_w > box_h:
            w_new = box_w * norm_ratio
            h_new = w_new
        else:
            h_new = box_h * norm_ratio
            w_new = h_new
    elif norm_method == "diagonal_square":
        diagonal = np.sqrt(np.pow(box_w, 2.0) + np.pow(box_h, 2.0))
        w_new = h_new = diagonal * norm_ratio
    elif norm_method in ["nothing", "none"]:
        pass

    bbox[0] = center_x - w_new / 2.0
    bbox[1] = center_y - h_new / 2.0
    bbox[2] = center_x + w_new / 2.0
    bbox[3] = center_y + h_new / 2.0

    # clip and record pad width
    expand = [0] * 4
    if bbox[0] < 0:
        expand[0] = bbox[0]
        bbox[0] = 0
    if bbox[1] < 0:
        expand[1] = bbox[1]
        bbox[1] = 0
    if bbox[2] > img_shape[1]:
        expand[2] = bbox[2] - img_shape[1]
        bbox[2] = img_shape[1]
    if bbox[3] > img_shape[0]:
        expand[3] = bbox[3] - img_shape[0]
        bbox[3] = img_shape[0]

    # to int
    bbox = list(map(int, bbox))
    expand = list(map(int, expand))

    return bbox, expand
