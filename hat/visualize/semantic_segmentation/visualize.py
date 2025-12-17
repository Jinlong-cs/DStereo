import base64
import os
from io import BytesIO

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa
from matplotlib.ticker import MultipleLocator  # noqa


def img_to_b64(img, ext=".png"):
    img_bin = cv2.imencode(ext, img)[1]
    img_bin_b64 = base64.b64encode(img_bin)
    return img_bin_b64


def base64_to_img(b64_str):
    bytes_str = base64.b64decode(b64_str)
    npar = np.frombuffer(bytes_str, dtype=np.uint8)
    img = cv2.imdecode(npar, cv2.IMREAD_UNCHANGED)
    return img


def plot_cm(classes, matrix):
    matrix = matrix.astype(np.float32)
    num_gts = matrix.sum(axis=1)
    num_prs = matrix.sum(axis=0)
    # Normalize by num of gts
    normed_matrix = matrix / num_gts.reshape(-1, 1)
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111)
    cax = ax.matshow(normed_matrix, cmap=plt.cm.viridis)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if np.isnan(normed_matrix[row, col]):
                ax.text(col, row, "NaN", va="center", ha="center", fontsize=12)
            else:
                ax.text(
                    col,
                    row,
                    "%.1f%%" % (normed_matrix[row, col] * 100),
                    va="center",
                    ha="center",
                    fontsize=12,
                )
    classes_with_num_gts = list(
        map(lambda c_n: "%s(%d)" % (c_n[0], c_n[1]), zip(classes, num_gts))
    )
    classes_with_num_prs = list(
        map(lambda c_n: "%s(%d)" % (c_n[0], c_n[1]), zip(classes, num_prs))
    )
    ax.set_xticklabels([None] + classes_with_num_prs, rotation=45)
    ax.set_yticklabels([None] + classes_with_num_gts)
    ax.set_xlabel("Predict", fontdict={"fontsize": 20})
    ax.set_ylabel("Groundtruth", fontdict={"fontsize": 20})

    b_io = BytesIO()
    fig.savefig(b_io, bbox_inches="tight")
    b_io.seek(0)
    plt.close(fig)
    return b_io


def draw_line(im, pts, color):
    for i in range(pts.shape[0] - 1):
        cv2.line(im, tuple(pts[i, :]), tuple(pts[i + 1, :]), color, 2)
    return im


def render_image(
    image,
    diff_map,
    gt_label_map,
    pred_label_map,
    gt_freespace_line=None,
    pred_freespace_line=None,
    labels_info=None,
    image_map=None,
    add_vis_info=None,
):
    raw_h, raw_w, raw_c = image.shape
    rendered_image = np.zeros((raw_h * 2, raw_w * 2, raw_c), np.uint8)
    rendered_image[:raw_h, :raw_w, :] = image

    h, w = diff_map.shape
    diff_image = np.zeros_like(image)
    sub_diff_image = diff_image[:h, :w]
    sub_diff_image[diff_map == 1] = (255, 255, 255)
    cv2.putText(
        diff_image,
        "DIFF",
        (22, 32),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale=w / 640.0,
        color=(0, 0, 0),
        thickness=2,
    )
    cv2.putText(
        diff_image,
        "DIFF",
        (20, 30),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale=w / 640.0,
        color=(255, 255, 255),
        thickness=2,
    )
    rendered_image[:raw_h, raw_w:, :] = diff_image

    h, w = gt_label_map.shape
    if image_map is not None:
        gt_mask = image_map.copy()
    else:
        gt_mask = image.copy()
    sub_gt_mask = gt_mask[:h, :w]
    for label_info in labels_info:
        if label_info.get("ignore", False):
            continue
        color_mode = label_info.get("color_mode", "bgr")
        if color_mode.lower() == "bgr":
            label_color = label_info["color"]
        elif color_mode.lower() == "rgb":
            label_color = label_info["color"][::-1]
        else:
            raise Exception("Invalid color mode")
        sub_gt_mask[gt_label_map == label_info["label"]] = label_color
    if add_vis_info is not None:
        for vis_info in add_vis_info:
            if color_mode.lower() == "bgr":
                vis_color = vis_info["color"]
            elif color_mode.lower() == "rgb":
                vis_color = vis_info["color"][::-1]
            sub_gt_mask[gt_label_map == vis_info["label"]] = vis_color
    if image_map is not None:
        gt_image = cv2.addWeighted(image_map, 0.6, gt_mask, 0.4, 0)
    else:
        gt_image = cv2.addWeighted(image, 0.6, gt_mask, 0.4, 0)
    cv2.putText(
        gt_image,
        "GT",
        (22, 32),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale=w / 640.0,
        color=(0, 0, 0),
        thickness=2,
    )
    cv2.putText(
        gt_image,
        "GT",
        (20, 30),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale=w / 640.0,
        color=(255, 255, 255),
        thickness=2,
    )
    rendered_image[raw_h:, :raw_w, :] = gt_image

    if image_map is not None:
        pred_mask = image_map.copy()
    else:
        pred_mask = image.copy()
    h, w = pred_label_map.shape
    sub_pred_mask = pred_mask[:h, :w]
    for label_info in labels_info:
        if label_info.get("ignore", False):
            continue
        color_mode = label_info.get("color_mode", "bgr")
        if color_mode.lower() == "bgr":
            label_color = label_info["color"]
        elif color_mode.lower() == "rgb":
            label_color = label_info["color"][::-1]
        else:
            raise Exception("Invalid color mode")
        sub_pred_mask[pred_label_map == label_info["label"]] = label_color
    if image_map is not None:
        pred_image = cv2.addWeighted(image_map, 0.6, pred_mask, 0.4, 0)
    else:
        pred_image = cv2.addWeighted(image, 0.6, pred_mask, 0.4, 0)
    if gt_freespace_line is not None and pred_freespace_line is not None:
        gt_freespace_line = np.array(gt_freespace_line)
        pred_freespace_line = np.array(pred_freespace_line)
        pts_gt = np.hstack(
            (
                np.array(range(len(gt_freespace_line))).reshape(-1, 1),
                gt_freespace_line.reshape(-1, 1),
            )
        )
        pts_pred = np.hstack(
            (
                np.array(range(len(pred_freespace_line))).reshape(-1, 1),
                pred_freespace_line.reshape(-1, 1),
            )
        )
        pts_gt = np.delete(pts_gt, np.where(pts_gt == -1)[0], axis=0)
        pts_pred = np.delete(pts_pred, np.where(pts_pred == -1)[0], axis=0)
        pred_image = draw_line(pred_image, pts_gt, [0, 255, 0])
        pred_image = draw_line(pred_image, pts_pred, [0, 0, 255])
    cv2.putText(
        pred_image,
        "PRED",
        (22, 32),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale=w / 640.0,
        color=(0, 0, 0),
        thickness=2,
    )
    cv2.putText(
        pred_image,
        "PRED",
        (20, 30),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale=w / 640.0,
        color=(255, 255, 255),
        thickness=2,
    )
    rendered_image[raw_h:, raw_w:, :] = pred_image
    return rendered_image


def save_eval_images(
    outfile_dir,
    image_name,
    diff,
    gt_label,
    pred_label,
    image,
):
    diff_dir = os.path.join(outfile_dir, "res_diff")
    if not os.path.exists(diff_dir):
        os.makedirs(diff_dir)
    gt_label_dir = os.path.join(outfile_dir, "res_gt_label")
    if not os.path.exists(gt_label_dir):
        os.makedirs(gt_label_dir)
    pred_label_dir = os.path.join(outfile_dir, "res_pred_label")
    if not os.path.exists(pred_label_dir):
        os.makedirs(pred_label_dir)
    image_dir = os.path.join(outfile_dir, "res_image")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    cv2.imwrite(
        os.path.join(diff_dir, os.path.splitext(image_name)[0] + ".png"),
        diff.astype("uint8"),
    )
    cv2.imwrite(
        os.path.join(gt_label_dir, os.path.splitext(image_name)[0] + ".png"),
        gt_label.astype("uint8"),
    )
    cv2.imwrite(
        os.path.join(
            pred_label_dir,
            os.path.splitext(image_name)[0] + ".png",
        ),
        pred_label.astype("uint8"),
    )
    cv2.imwrite(os.path.join(image_dir, image_name), image)


def render_class_image(
    image,
    diff_map,
    gt_label_map,
    pred_label_map,
    labels_info,
    iou_value,
    image_map=None,
    add_vis_info=None,
):
    raw_h, raw_w, raw_c = image.shape
    rendered_image = np.zeros((raw_h * 2, raw_w * 2, raw_c), np.uint8)
    rendered_image[:raw_h, :raw_w, :] = image

    h, w = diff_map.shape
    diff_image = np.zeros_like(image)
    sub_diff_image = diff_image[:h, :w]
    sub_diff_image[diff_map == 1] = (255, 255, 255)
    cv2.putText(
        diff_image,
        "DIFF",
        (22, 32),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale=w / 640.0,
        color=(0, 0, 0),
        thickness=2,
    )
    cv2.putText(
        diff_image,
        "DIFF",
        (20, 30),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale=w / 640.0,
        color=(255, 255, 255),
        thickness=2,
    )
    rendered_image[:raw_h, raw_w:, :] = diff_image

    h, w = gt_label_map.shape
    if image_map is not None:
        gt_mask = image_map.copy()
    else:
        gt_mask = image.copy()
    sub_gt_mask = gt_mask[:h, :w]
    for label_info in labels_info:
        if label_info.get("ignore", False):
            continue
        color_mode = label_info.get("color_mode", "bgr")
        if color_mode.lower() == "bgr":
            label_color = label_info["color"]
        elif color_mode.lower() == "rgb":
            label_color = label_info["color"][::-1]
        else:
            raise Exception("Invalid color mode")
        sub_gt_mask[gt_label_map == label_info["label"]] = label_color
    if add_vis_info is not None:
        for vis_info in add_vis_info:
            if color_mode.lower() == "bgr":
                vis_color = vis_info["color"]
            elif color_mode.lower() == "rgb":
                vis_color = vis_info["color"][::-1]
            sub_gt_mask[gt_label_map == vis_info["label"]] = vis_color
    if image_map is not None:
        gt_image = cv2.addWeighted(image_map, 0.6, gt_mask, 0.4, 0)
    else:
        gt_image = cv2.addWeighted(image, 0.6, gt_mask, 0.4, 0)
    cv2.putText(
        gt_image,
        "GT",
        (22, 32),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale=w / 640.0,
        color=(0, 0, 0),
        thickness=2,
    )
    cv2.putText(
        gt_image,
        "GT",
        (20, 30),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale=w / 640.0,
        color=(255, 255, 255),
        thickness=2,
    )
    rendered_image[raw_h:, :raw_w, :] = gt_image

    if image_map is not None:
        pred_mask = image_map.copy()
    else:
        pred_mask = image.copy()
    h, w = pred_label_map.shape
    sub_pred_mask = pred_mask[:h, :w]
    for label_info in labels_info:
        if label_info.get("ignore", False):
            continue
        color_mode = label_info.get("color_mode", "bgr")
        if color_mode.lower() == "bgr":
            label_color = label_info["color"]
        elif color_mode.lower() == "rgb":
            label_color = label_info["color"][::-1]
        else:
            raise Exception("Invalid color mode")
        sub_pred_mask[pred_label_map == label_info["label"]] = label_color
    if image_map is not None:
        pred_image = cv2.addWeighted(image_map, 0.6, pred_mask, 0.4, 0)
    else:
        pred_image = cv2.addWeighted(image, 0.6, pred_mask, 0.4, 0)
    cv2.putText(
        pred_image,
        "PRED",
        (22, 32),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale=w / 640.0,
        color=(0, 0, 0),
        thickness=2,
    )
    cv2.putText(
        pred_image,
        "PRED",
        (20, 30),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale=w / 640.0,
        color=(255, 255, 255),
        thickness=2,
    )
    cv2.putText(
        pred_image,
        "IOU: %.4f" % iou_value,
        (22, 70),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale=w / 640.0,
        color=(0, 0, 0),
        thickness=2,
    )
    cv2.putText(
        pred_image,
        "IOU: %.4f" % iou_value,
        (20, 68),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale=w / 640.0,
        color=(255, 255, 255),
        thickness=2,
    )
    rendered_image[raw_h:, raw_w:, :] = pred_image
    return rendered_image


def render_model(
    image,
    diff_map,
    pred_label_map_m1,
    pred_label_map_m2,
    labels_info_m1,
    labels_info_m2,
    pred_1_name,
    pred_2_name,
    image_1_map=None,
    image_2_map=None,
):
    raw_h, raw_w, raw_c = image.shape
    rendered_image = np.zeros((raw_h * 2, raw_w * 2, raw_c), np.uint8)
    rendered_image[:raw_h, :raw_w, :] = image

    h, w = diff_map.shape
    diff_image = np.zeros_like(image)
    sub_diff_image = diff_image[:h, :w]
    sub_diff_image[diff_map == 1] = (255, 255, 255)
    cv2.putText(
        diff_image,
        "DIFF",
        (22, 32),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale=w / 640.0,
        color=(0, 0, 0),
        thickness=2,
    )
    cv2.putText(
        diff_image,
        "DIFF",
        (20, 30),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale=w / 640.0,
        color=(255, 255, 255),
        thickness=2,
    )
    rendered_image[:raw_h, raw_w:, :] = diff_image

    h, w = pred_label_map_m1.shape
    if image_1_map is not None:
        gt_mask = image_1_map.copy()
    else:
        gt_mask = image.copy()
    sub_gt_mask = gt_mask[:h, :w]
    for label_info in labels_info_m1:
        if label_info.get("ignore", False):
            continue
        color_mode = label_info.get("color_mode", "bgr")
        if color_mode.lower() == "bgr":
            label_color = label_info["color"]
        elif color_mode.lower() == "rgb":
            label_color = label_info["color"][::-1]
        else:
            raise Exception("Invalid color mode")
        sub_gt_mask[pred_label_map_m1 == label_info["label"]] = label_color
    if image_1_map is not None:
        gt_image = cv2.addWeighted(image_1_map, 0.6, gt_mask, 0.4, 0)
    else:
        gt_image = cv2.addWeighted(image, 0.6, gt_mask, 0.4, 0)
    cv2.putText(
        gt_image,
        pred_1_name,
        (22, 32),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale=w / 640.0,
        color=(0, 0, 0),
        thickness=2,
    )
    cv2.putText(
        gt_image,
        pred_1_name,
        (20, 30),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale=w / 640.0,
        color=(255, 255, 255),
        thickness=2,
    )
    rendered_image[raw_h:, :raw_w, :] = gt_image

    if image_2_map is not None:
        pred_mask = image_2_map.copy()
    else:
        pred_mask = image.copy()
    h, w = pred_label_map_m2.shape
    sub_pred_mask = pred_mask[:h, :w]
    for label_info in labels_info_m2:
        if label_info.get("ignore", False):
            continue
        color_mode = label_info.get("color_mode", "bgr")
        if color_mode.lower() == "bgr":
            label_color = label_info["color"]
        elif color_mode.lower() == "rgb":
            label_color = label_info["color"][::-1]
        else:
            raise Exception("Invalid color mode")
        sub_pred_mask[pred_label_map_m2 == label_info["label"]] = label_color
    if image_2_map is not None:
        pred_image = cv2.addWeighted(image_2_map, 0.6, pred_mask, 0.4, 0)
    else:
        pred_image = cv2.addWeighted(image, 0.6, pred_mask, 0.4, 0)
    cv2.putText(
        pred_image,
        pred_2_name,
        (22, 32),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale=w / 640.0,
        color=(0, 0, 0),
        thickness=2,
    )
    cv2.putText(
        pred_image,
        pred_2_name,
        (20, 30),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale=w / 640.0,
        color=(255, 255, 255),
        thickness=2,
    )
    rendered_image[raw_h:, raw_w:, :] = pred_image
    return rendered_image
