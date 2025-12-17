# Copyright (c) Horizon Robotics. All rights reserved.
# Get roi according to setting. And the metric is calculated
# based on roi.
from collections import OrderedDict

import cv2
import numpy as np

from .camera import FisheyeCamera as Camera


def draw_vp(image, vp_x, vp_y, size=20, color=(0, 0, 255), thickness=2):
    cv2.line(
        image,
        (int(vp_x - size / 2), vp_y),
        (int(vp_x + size / 2), vp_y),
        color=color,
        thickness=thickness,
    )
    cv2.line(
        image,
        (vp_x, int(vp_y - size / 2)),
        (vp_x, int(vp_y + size / 2)),
        color=color,
        thickness=thickness,
    )
    return


def get_pts_from_ends(
    image_width,
    image_height,
    end_pt1,
    end_pt2,
    vp_x=0,
    vp_y=0,
    add_height=1000,
    is_fisheye_camera=False,
):
    if (end_pt1[1] > 0) and (end_pt1[1] < vp_y):
        y = image_height + add_height
        x = (
            int(
                (end_pt2[0] - end_pt1[0])
                / (end_pt2[1] - end_pt1[1])
                * (y - end_pt1[1])
            )
            + end_pt1[0]
        )
        end_pt1[0] = x
        end_pt1[1] = y
    if (end_pt2[1] > 0) and (end_pt2[1] < vp_y):
        y = image_height + add_height
        x = (
            int(
                (end_pt2[0] - end_pt1[0])
                / (end_pt2[1] - end_pt1[1])
                * (y - end_pt1[1])
            )
            + end_pt1[0]
        )
        end_pt2[0] = x
        end_pt2[1] = y
    if not is_fisheye_camera:
        if end_pt1[1] < 0:
            end_pt1[1] = image_height
        if end_pt2[1] < 0:
            end_pt2[1] = image_height
    result_pts = []
    result_pts.append(end_pt1)
    if abs(end_pt1[0] - end_pt2[0]) >= abs(end_pt1[1] - end_pt2[1]):
        if abs(end_pt1[0] - end_pt2[0]) < 2:
            result_pts.append(end_pt2)
            return result_pts
        for x in range(end_pt1[0] + 1, end_pt2[0]):
            if end_pt2[1] - end_pt1[1] == 0:
                y = end_pt2[1]
            else:
                y = max(
                    0,
                    min(
                        image_height,
                        int(
                            (end_pt2[1] - end_pt1[1])
                            / (end_pt2[0] - end_pt1[0])
                            * (x - end_pt1[0])
                        )
                        + end_pt1[1],
                    ),
                )
            x = max(min(x, image_width), 0)
            result_pts.append([x, y])
    else:
        if abs(end_pt1[1] - end_pt2[1]) < 2:
            result_pts.append(end_pt2)
            return result_pts
        for y in range(end_pt1[1] + 1, end_pt2[1]):
            if end_pt2[0] - end_pt1[0] == 0:
                x = end_pt2[0]
            else:
                x = max(
                    0,
                    min(
                        image_width,
                        int(
                            (end_pt2[0] - end_pt1[0])
                            / (end_pt2[1] - end_pt1[1])
                            * (y - end_pt1[1])
                        )
                        + end_pt1[0],
                    ),
                )
            y = max(min(y, image_height), 0)
            result_pts.append([x, y])

    result_pts.append(end_pt2)
    return result_pts


def get_point_bound(
    image_width,
    image_height,
    pre_end_points,
    new_end_points,
    camera,
    use_distort=False,
    is_fisheye_camera=False,
):
    bound_points_list = list()  # noqa: C408
    pre_end_pt1 = pre_end_points[0]
    pre_end_pt2 = pre_end_points[1]
    new_end_pt1 = new_end_points[0]
    new_end_pt2 = new_end_points[1]
    bound_points_list.extend(
        get_pts_from_ends(
            image_width,
            image_height,
            pre_end_pt2,
            new_end_pt2,
            camera.vp_x,
            camera.vp_y,
            is_fisheye_camera=is_fisheye_camera,
        )
    )
    bound_points_list.extend(
        get_pts_from_ends(
            image_width,
            image_height,
            new_end_pt2,
            new_end_pt1,
            camera.vp_x,
            camera.vp_y,
            is_fisheye_camera=is_fisheye_camera,
        )
    )
    bound_points_list.extend(
        get_pts_from_ends(
            image_width,
            image_height,
            new_end_pt1,
            pre_end_pt1,
            camera.vp_x,
            camera.vp_y,
            is_fisheye_camera=is_fisheye_camera,
        )
    )
    bound_points_list.extend(
        get_pts_from_ends(
            image_width,
            image_height,
            pre_end_pt1,
            pre_end_pt2,
            camera.vp_x,
            camera.vp_y,
            is_fisheye_camera=is_fisheye_camera,
        )
    )
    if use_distort:
        ret_points = get_distort_points(
            camera,
            bound_points_list,
            image_width,
            image_height,
        )
        return ret_points
    return bound_points_list


def get_distort_points(camera, points_list, image_width, image_height):
    distorts_list = list()  # noqa: C408
    for pts in points_list:
        if pts[0] == 0 or pts[0] == image_width or pts[1] == image_height:
            distorts_list.append(pts)
        else:
            xp, yp = camera.get_distort_points(pts[0], pts[1])
            distorts_list.append(
                [max(0, min(image_width, xp)), max(0, min(image_height, yp))]
            )
    return distorts_list


def get_image_laterallongti_distance_v2(
    image_anno,
    longi_distance_list,
    lateral_distance_list,
    lateral_add_zero=True,
    is_fisheye_camera=False,
    edge_add_value=1000,
):
    img_name = image_anno["image_name"]
    img_height = image_anno["image_height"]
    img_width = image_anno["image_width"]
    camera_param = image_anno["camera_param"]
    camera = Camera(camera_param, is_fisheye_camera=is_fisheye_camera)
    return_longi_distance_image_dict = OrderedDict()
    return_lat_distance_image_dict = OrderedDict()
    return_longi_lateral_distance_image_roi_dict = OrderedDict()
    if longi_distance_list is None and lateral_distance_list is None:
        return (
            img_name,
            img_height,
            img_width,
            return_longi_lateral_distance_image_roi_dict,
            camera,
        )
    if (longi_distance_list is not None) and (len(longi_distance_list) > 1):
        for longi_dis in longi_distance_list:
            if longi_dis > 40:
                lat_x = 6
            else:
                lat_x = 1
            left_pt = camera.get_image_point_vcs2image(longi_dis, -lat_x)
            right_pt = camera.get_image_point_vcs2image(longi_dis, lat_x)
            return_longi_distance_image_dict[longi_dis] = [left_pt, right_pt]
    if (
        (lateral_distance_list is not None)
        and (len(lateral_distance_list) > 0)
        and (len(lateral_distance_list[0]) > 0)
    ):
        for lat_list in lateral_distance_list:
            if lat_list[1] < 3:
                y_far, y_close = 80, 60
            else:
                y_far, y_close = 100, 80
            left_line_far_pt = camera.get_image_point_vcs2image(
                y_far, lat_list[1]
            )
            left_line_close_pt = camera.get_image_point_vcs2image(
                y_close, lat_list[1]
            )
            right_line_far_pt = camera.get_image_point_vcs2image(
                y_far, lat_list[0]
            )
            right_line_close_pt = camera.get_image_point_vcs2image(
                y_close, lat_list[0]
            )
            return_lat_distance_image_dict[lat_list[1]] = [
                [left_line_close_pt, left_line_far_pt],
                [right_line_close_pt, right_line_far_pt],
            ]
    if lateral_add_zero:
        lateral_distance_list.insert(0, [0, 0])
        return_lat_distance_image_dict[0] = [
            [[-edge_add_value, 200], [-edge_add_value, 100]],
            [
                [img_width + edge_add_value, 200],
                [img_width + edge_add_value, 100],
            ],
        ]
    return_longi_lateral_distance_image_roi_dict = get_undistort_eval_roi(
        camera,
        img_width,
        img_height,
        longi_distance_list,
        lateral_distance_list,
        return_longi_distance_image_dict,
        return_lat_distance_image_dict,
        is_vcs_coord=True,
    )
    return (
        img_name,
        img_height,
        img_width,
        return_longi_lateral_distance_image_roi_dict,
        camera,
    )


def get_image_laterallongti_distance_fromLocal(
    image_anno,
    longi_distance_list,
    lateral_distance_list,
    lateral_add_zero=True,
    is_fisheye_camera=False,
    edge_add_value=1000,
):
    # img_name = image_anno['frame_name']
    img_name = image_anno["image_name"]
    img_height = image_anno["image_height"]
    img_width = image_anno["image_width"]
    camera_param = image_anno["camera_param"]
    # camera_param_str = image_anno['camera_param']
    # camera_param = json.loads(camera_param_str)
    camera = Camera(camera_param, is_fisheye_camera=is_fisheye_camera)

    if camera.invalid_camera is True:
        return None, None, None, None, None
    return_longi_distance_image_dict = OrderedDict()
    return_lat_distance_image_dict = OrderedDict()
    return_longi_lateral_distance_image_roi_dict = OrderedDict()
    if longi_distance_list is None and lateral_distance_list is None:
        return (
            img_name,
            img_height,
            img_width,
            return_longi_lateral_distance_image_roi_dict,
            camera,
        )
    if (longi_distance_list is not None) and (len(longi_distance_list) > 1):
        for longi_dis in longi_distance_list:
            if longi_dis > 40:
                lat_x = 6
            else:
                lat_x = 1
            left_pt = camera.get_image_point_local2image(longi_dis, -lat_x)
            right_pt = camera.get_image_point_local2image(longi_dis, lat_x)
            return_longi_distance_image_dict[longi_dis] = [left_pt, right_pt]
    if (
        (lateral_distance_list is not None)
        and (len(lateral_distance_list) > 0)
        and (len(lateral_distance_list[0]) > 0)
    ):
        for lat_list in lateral_distance_list:
            if lat_list[1] < 3:
                y_far, y_close = 40, 20
            else:
                y_far, y_close = 60, 30
            left_line_far_pt = camera.get_image_point_local2image(
                y_far, lat_list[1]
            )
            left_line_close_pt = camera.get_image_point_local2image(
                y_close, lat_list[1]
            )
            right_line_far_pt = camera.get_image_point_local2image(
                y_far, lat_list[0]
            )
            right_line_close_pt = camera.get_image_point_local2image(
                y_close, lat_list[0]
            )
            return_lat_distance_image_dict[lat_list[1]] = [
                [left_line_close_pt, left_line_far_pt],
                [right_line_close_pt, right_line_far_pt],
            ]
    # add -1
    if lateral_add_zero:
        lateral_distance_list.insert(0, [0, 0])
        return_lat_distance_image_dict[0] = [
            [[-edge_add_value, 200], [-edge_add_value, 100]],
            [
                [img_width + edge_add_value, 200],
                [img_width + edge_add_value, 100],
            ],
        ]
    return_longi_lateral_distance_image_roi_dict = get_undistort_eval_roi(
        camera,
        img_width,
        img_height,
        longi_distance_list,
        lateral_distance_list,
        return_longi_distance_image_dict,
        return_lat_distance_image_dict,
        is_vcs_coord=False,
    )
    return (
        img_name,
        img_height,
        img_width,
        return_longi_lateral_distance_image_roi_dict,
        camera,
    )


def get_special_point_xsame(x0_1, y0_1, x0_2, y0_2, x3_1, y3_1, x4_1, y4_1):
    if x3_1 == x4_1:
        return list()  # noqa: C408
    elif y3_1 == y4_1:
        return [x0_1, y3_1]
    else:
        x = x0_1
        y = int(
            (y4_1 - y3_1) * x / (x4_1 - x3_1)
            + (y3_1 * x4_1 - y4_1 * x3_1) / (x4_1 - x3_1)
        )
        return [x, y]


# y0_1 == y0_2
def get_special_point_ysame(x0_1, y0_1, x0_2, y0_2, x3_1, y3_1, x4_1, y4_1):
    if y3_1 == y4_1:
        return list()  # noqa: C408
    elif x3_1 == x4_1:
        return [x3_1, y0_1]
    else:
        y = y0_1
        x = int(
            (x4_1 - x3_1) * y / (y4_1 - y3_1)
            - (y3_1 * x4_1 - y4_1 * x3_1) / (y4_1 - y3_1)
        )
        return [x, y]


def get_twolines_Intersec(line1_2pts_list, line2_2pts_list):
    x1, y1 = line1_2pts_list[0]
    x2, y2 = line1_2pts_list[1]
    x3, y3 = line2_2pts_list[0]
    x4, y4 = line2_2pts_list[1]
    if x1 == x2:
        insec_pt = get_special_point_xsame(x1, y1, x2, y2, x3, y3, x4, y4)
    elif x3 == x4:
        insec_pt = get_special_point_xsame(x3, y3, x4, y4, x1, y1, x2, y2)
    elif y1 == y2:
        insec_pt = get_special_point_ysame(x1, y1, x2, y2, x3, y3, x4, y4)
    elif y3 == y4:
        insec_pt = get_special_point_ysame(x3, y3, x4, y4, x1, y1, x2, y2)
    else:
        k1 = (y2 - y1) / (x2 - x1)
        b1 = (y1 * x2 - y2 * x1) / (x2 - x1)
        k2 = (y4 - y3) / (x4 - x3)
        b2 = (y3 * x4 - y4 * x3) / (x4 - x3)
        insec_x = int((b2 - b1) / (k1 - k2))
        insec_y = int((b2 * k1 - b1 * k2) / (k1 - k2))
        insec_pt = [insec_x, insec_y]
    return insec_pt


def get_undistort_eval_roi(
    camera,
    image_width,
    image_height,
    longi_distance_list,
    lateral_distance_list,
    return_longi_distance_image_dict,
    return_lat_distance_image_dict,
    is_vcs_coord,
):
    return_longi_lateral_distance_image_dict = OrderedDict()
    if (
        (longi_distance_list is not None)
        and (len(longi_distance_list) > 1)
        and (lateral_distance_list is not None)
        and (len(lateral_distance_list[0]) > 0)
    ):
        for longi_dis in longi_distance_list:
            return_longi_lateral_distance_image_dict[longi_dis] = OrderedDict()
            line_longi = return_longi_distance_image_dict[longi_dis]
            for lat_dis_list in lateral_distance_list:
                lat_dis = lat_dis_list[1]
                left_lat_line = return_lat_distance_image_dict[lat_dis][0]
                right_lat_line = return_lat_distance_image_dict[lat_dis][1]
                if (
                    (line_longi[0][1] < 0)
                    or (line_longi[1][1] < 0)
                    or (line_longi[0][1] > image_height)
                    or (line_longi[1][1] > image_height)
                ):
                    line_longi = [
                        [-50, image_height],
                        [image_width + 50, image_height],
                    ]
                if lat_dis == 0:
                    if is_vcs_coord:
                        long_h = camera.get_image_height_vcs(longi_dis)
                    else:
                        long_h = camera.get_image_height_local(longi_dis)
                    insec_left_pt = [0, long_h]
                    insec_right_pt = [image_width, long_h]
                else:
                    insec_left_pt = get_twolines_Intersec(
                        line_longi, left_lat_line
                    )
                    insec_right_pt = get_twolines_Intersec(
                        line_longi, right_lat_line
                    )
                lateral_dis_dict = OrderedDict()
                lateral_dis_dict[lat_dis] = [insec_left_pt, insec_right_pt]
                return_longi_lateral_distance_image_dict[longi_dis].update(
                    lateral_dis_dict
                )
    return return_longi_lateral_distance_image_dict


def roi_valid(images_info_dict, roi_dict):
    if "horizon" in roi_dict:
        longitudinal_distance_setting = roi_dict["horizon"]
    else:
        longitudinal_distance_setting = None
    if "lateral" in roi_dict:
        lateral_distance_setting = [roi_dict["lateral"]]
        if len(lateral_distance_setting[0]) == 0:
            lateral_distance_setting = None
    else:
        lateral_distance_setting = None

    is_LocalCoord = roi_dict["is_local_coord"]
    is_fisheye_camera = images_info_dict["camera_param"]["fov"] > 150
    lateral_add_zero = roi_dict.get("lateral_add_zero", False)
    if is_LocalCoord:
        (
            frame_name,
            image_height,
            image_width,
            longi_lateral_distance_image_dict,
            image_camera,
        ) = get_image_laterallongti_distance_fromLocal(
            images_info_dict,
            longitudinal_distance_setting,
            lateral_distance_setting,
            lateral_add_zero=lateral_add_zero,
            is_fisheye_camera=is_fisheye_camera,
        )
    else:
        (
            frame_name,
            image_height,
            image_width,
            longi_lateral_distance_image_dict,
            image_camera,
        ) = get_image_laterallongti_distance_v2(
            images_info_dict,
            longitudinal_distance_setting,
            lateral_distance_setting,
            lateral_add_zero=lateral_add_zero,
            is_fisheye_camera=is_fisheye_camera,
        )
    if image_camera is None:
        return None, None, None
    longi_lat_points_dict = OrderedDict()
    if (longitudinal_distance_setting is not None) and (
        lateral_distance_setting is not None
    ):
        for longi_dis_index in range(len(longitudinal_distance_setting) - 1):
            longi_pre = longitudinal_distance_setting[longi_dis_index]
            lat_dict_pre = longi_lateral_distance_image_dict[longi_pre]
            longi_new = longitudinal_distance_setting[longi_dis_index + 1]
            lat_dict_new = longi_lateral_distance_image_dict[longi_new]
            lat_dis_points_dict = OrderedDict()
            for lat_dis_list in lateral_distance_setting:
                lat_dis = lat_dis_list[1]
                lat_dis_points_dict[lat_dis] = []
                lat_dis_points_dict[lat_dis].extend(
                    get_point_bound(
                        image_width,
                        image_height,
                        lat_dict_pre[lat_dis],
                        lat_dict_new[lat_dis],
                        image_camera,
                        roi_dict["distort"],
                        is_fisheye_camera=is_fisheye_camera,
                    )
                )
            longi_lat_points_dict[
                str(longi_pre) + "m-" + str(longi_new) + "m"
            ] = lat_dis_points_dict
    longitudinal_distance_setting.sort()
    label_index = 1
    lat_dis = lateral_distance_setting[0][1]
    longi_labelindex_dict = OrderedDict()
    longi_pre = longitudinal_distance_setting[0]
    longi_next = longitudinal_distance_setting[1]
    longi_key = str(longi_pre) + "m-" + str(longi_next) + "m"
    longi_labelindex_dict[longi_key] = label_index
    lat_points_dict = longi_lat_points_dict[longi_key]
    points = lat_points_dict[lat_dis]
    mask_array = np.zeros((image_height, image_width), np.uint8)
    cv2.fillConvexPoly(mask_array, np.asarray(points), label_index, 8)
    return mask_array, label_index, image_camera
