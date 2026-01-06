import os
import sys
import re
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as RR
import time
from datetime import datetime
import copy
import tqdm
from scipy.spatial.transform import Slerp


def interpolate_translation(t1, trans1, t2, trans2, t_inter):
    return trans1 + (trans2 - trans1) / (t2 - t1) * (t_inter - t1)


def find_closest_timestamps(data_dict, target_timestamp, start_=0):
    target_datetime = datetime.fromtimestamp(target_timestamp)
    history_clost_pose = None
    history_clost_pose_idx = None
    future_clost_pose = None
    future_clost_pose_idx = None
    for idx, pose in enumerate(data_dict[start_:]):
        pose_time = datetime.fromtimestamp(np.float64(list(pose.keys())[0]))
        if target_datetime == pose_time:
            return (
                True,
                list(data_dict[idx].values())[0],
                idx,
                list(data_dict[idx].values())[0],
                idx,
                0.0,
            )
        elif target_datetime > pose_time:
            if history_clost_pose is None:
                history_clost_pose = pose
                history_clost_pose_idx = idx
            else:
                if abs(target_datetime - pose_time) < abs(
                    target_datetime
                    - datetime.fromtimestamp(
                        np.float64(list(history_clost_pose.keys())[0])
                    )
                ):
                    history_clost_pose = pose
                    history_clost_pose_idx = idx
        else:
            if future_clost_pose is None:
                if history_clost_pose is None:
                    return False, None, None, None, None, None
                future_clost_pose = pose
                future_clost_pose_idx = idx
            return (
                True,
                list(history_clost_pose.values())[0],
                history_clost_pose_idx,
                list(future_clost_pose.values())[0],
                future_clost_pose_idx,
                np.float64(list(future_clost_pose.keys())[0])
                - np.float64(list(history_clost_pose.keys())[0]),
            )

    if history_clost_pose is None or future_clost_pose is None:
        return False, None, None, None, None, None
    return (
        True,
        list(history_clost_pose.values())[0],
        history_clost_pose_idx,
        list(future_clost_pose.values())[0],
        future_clost_pose_idx,
        np.float64(list(future_clost_pose.keys())[0])
        - np.float64(list(history_clost_pose.keys())[0]),
    )


def transform_point(raw_points, transformation_matrix):
    points = raw_points[:, :3]
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    points = np.matmul(transformation_matrix, points.T).T
    raw_points[:, :3] = points[:, :3]

    return raw_points


def pose2rotation(min_pose):
    trans1 = np.array([float(i) for i in min_pose[1:4]])
    rotation = np.eye(4)
    rotation[:3, :3] = RR.from_quat(min_pose[-4:]).as_matrix()
    rotation[0, 3] = trans1[0]
    rotation[1, 3] = trans1[1]
    rotation[2, 3] = trans1[2]

    return rotation


def rotation2pose(rotation):
    quat = RR.from_matrix(rotation[:3, :3]).as_quat()
    pose = [*rotation[0:3, 3]] + list(quat)

    return pose


def load_pose(pose_file):
    pose_dict = []
    with open(pose_file, "r") as file:
        lines = file.readlines()
    for line in lines:
        txt_info = line.strip("\n").split(" ")
        pose_dict.append({txt_info[0]: txt_info})
    return pose_dict


def load_imu(imu_file):
    pose_dict = []
    max_yaw_speed = 0
    with open(imu_file, "r") as file:
        lines = file.readlines()
    for line in lines:
        pose_dict.append(line.strip("\n").split(" "))
        if abs(np.float64(pose_dict[-1][2])) > max_yaw_speed:
            max_yaw_speed = abs(np.float64(pose_dict[-1][2]))
    print("max yaw speed: ", max_yaw_speed)
    return pose_dict


def odom_filter(imu_info, target_timestamp):
    for imu_ in imu_info.values():
        imu_time = np.float64(imu_[0])
        if target_timestamp > imu_time:
            continue
        return abs(np.array(imu_[1:], dtype=np.float32)).max()
    return False


def imu_filter(imu_info, target_timestamp):
    min_time_diff = 1000
    rtn_imu = None
    for imu_ in imu_info:
        imu_time = np.float64(imu_[0])
        if target_timestamp >= imu_time:
            if (target_timestamp - imu_time) < min_time_diff:
                min_time_diff = target_timestamp - imu_time
                rtn_imu = imu_
        else:
            if abs(target_timestamp - imu_time) < min_time_diff:
                min_time_diff = abs(target_timestamp - imu_time)
                rtn_imu = imu_
            break
    if rtn_imu is None:
        return False

    return abs(np.array(rtn_imu[1:4], dtype=np.float64)).max()


def load_odom(pose_file):
    pose_dict = {}
    with open(pose_file, "r") as file:
        lines = file.readlines()
    for line in lines:
        txt_info = line.strip("\n").split(" ")
        pose_dict.update({txt_info[0]: txt_info})
    return pose_dict


def load_pcd(pcd_path):
    lines = []
    num_points = None

    with open(pcd_path, "r") as f:
        for line in f:
            lines.append(line.strip())
            if line.startswith("POINTS"):
                num_points = int(line.split()[-1])
    assert num_points is not None

    points = []
    for line in lines[-num_points:]:
        x, y, z = list(map(float, line.split()))
        points.append([x, y, z])

    return np.array(points)


def search_lidar_path_by_pose_timest(lidar_root, pose_time):
    lidar_path = None
    lidar_path_time = None
    all_lidar_list = os.listdir(lidar_root)
    all_lidar_list.sort()
    for idx, lidar_file in enumerate(all_lidar_list):
        lidar_time = np.float64(lidar_file[:-4])
        if lidar_time < pose_time:
            continue
        elif lidar_time == pose_time:
            lidar_path = lidar_file
            lidar_path_time = lidar_time
            break
        else:
            if idx == 0:
                lidar_path = lidar_file
                lidar_path_time = lidar_time
                break
            lidar_path = lidar_file
            lidar_path_time = lidar_time
            pre_lidar_time = np.float64(all_lidar_list[idx - 1][:-4])
            pre_target_time_diff = abs(pose_time - pre_lidar_time)
            if pre_target_time_diff < (lidar_time - pose_time):
                lidar_path = all_lidar_list[idx - 1]
                lidar_path_time = pre_lidar_time
            break
    if lidar_path is None:
        return False, 0.0
    return os.path.join(lidar_root, lidar_path), lidar_path_time


def interploate_pose(pre_pose, next_pose, time_inter, return_pose=False):
    slerp_func = Slerp(
        [np.float64(pre_pose[0]), np.float64(next_pose[0])],
        RR.from_quat([pre_pose[-4:], next_pose[-4:]]),
    )

    trans1 = np.array([float(i) for i in pre_pose[1:4]])
    trans2 = np.array([float(i) for i in next_pose[1:4]])
    trans_inter = interpolate_translation(
        np.float64(pre_pose[0]), trans1, np.float64(next_pose[0]), trans2, time_inter
    )
    if return_pose:
        return [time_inter, *trans_inter, *slerp_func(time_inter).as_quat()]
    rotation = np.eye(4)
    rotation[:3, :3] = slerp_func(time_inter).as_matrix()
    rotation[0, 3] = trans_inter[0]
    rotation[1, 3] = trans_inter[1]
    rotation[2, 3] = trans_inter[2]
    return rotation


def replace_norm_pose_to_keyframe(norm_pose, key_frame_pose_dict):
    key_frame_pose_time_list = [list(i.values())[0] for i in key_frame_pose_dict]
    for idx, key_frame_pose_time in enumerate(key_frame_pose_time_list):
        if np.float64(norm_pose[0]) == np.float64(key_frame_pose_time[0]):
            return True, key_frame_pose_time
        if np.float64(key_frame_pose_time[0]) > np.float64(norm_pose[0]):
            break
    return False, norm_pose


def load_pose_revise(pose_file, key_frame_pose):
    pose_dict = []
    with open(pose_file, "r") as file:
        lines = file.readlines()
    pre_pose = None
    pre_pose_revise = None
    for line in lines:
        txt_info = line.strip("\n").split(" ")
        flag, current_pose = replace_norm_pose_to_keyframe(
            copy.deepcopy(txt_info), key_frame_pose
        )
        if not flag:
            if len(pose_dict) == 0:
                continue
            current_pose = (
                pose2rotation(copy.deepcopy(txt_info))
                @ np.linalg.inv(pose2rotation(pre_pose))
                @ pose2rotation(copy.deepcopy(pre_pose_revise))
            )
            current_pose = rotation2pose(current_pose)
            current_pose = [
                txt_info[0],
            ] + current_pose
        pose_dict.append({txt_info[0]: current_pose})
        pre_pose = copy.deepcopy(txt_info)
        pre_pose_revise = copy.deepcopy(current_pose)
    return pose_dict


def depth2disp(depth, baseline_rectified, f_rectified, verbose=False):
    depth[np.isnan(depth)] = 0
    depth[np.isinf(depth)] = 0

    mask = depth <= 0
    disp = (baseline_rectified * f_rectified) / (depth + 1e-6)
    disp[mask] = 0
    if verbose:
        mask_dis = disp >= 1.0
        print(
            "disparity density: %d / %d = %0.2f"
            % (
                mask_dis.astype(np.int32).sum(),
                depth.size,
                float(mask_dis.astype(np.int32).sum()) / depth.size,
            )
        )
    assert (disp == 0).sum() == (depth == 0).sum()
    return disp.astype(np.float32)


def disp2depth(depth, baseline_rectified, f_rectified, disp_max, disp_min):
    disp = (baseline_rectified * f_rectified) / (depth + 1e-6)
    mask = depth < disp_min
    disp[mask] = 0
    mask = depth > disp_max
    disp[mask] = 0
    return disp


def readPFM(file):
    file = open(file, "rb")

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if (sys.version[0]) == "3":
        header = header.decode("utf-8")
    if header == "PF":
        color = True
    elif header == "Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    if (sys.version[0]) == "3":
        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
    else:
        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    if (sys.version[0]) == "3":
        scale = float(file.readline().rstrip().decode("utf-8"))
    else:
        scale = float(file.readline().rstrip())

    if scale < 0:
        endian = "<"
        scale = -scale
    else:
        endian = ">"

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def disp2rgb(disp, disp_max, disp_min):
    mask = np.logical_or(disp > disp_max, disp < disp_min)
    disp = np.clip(disp, disp_min, disp_max)
    mat_min = disp_min
    mat_max = disp_max
    norm_matrix = (disp - mat_min) / (mat_max - mat_min)
    disp = 0.1 + norm_matrix * (0.9 - 0.1)
    disp *= 255
    disp = np.round(disp).astype(np.uint8)[..., None]
    disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
    disp[mask] = (0, 0, 0)

    return disp


def disp2gray(disp, disp_max, disp_min):
    mask = np.logical_or(disp > disp_max, disp < disp_min)
    disp = np.clip(disp, disp_min, disp_max)
    disp = disp / disp_max
    disp *= 255
    disp = np.round(disp).astype(np.uint8)[..., None]
    disp = cv2.applyColorMap(disp, cv2.COLORMAP_BONE)
    disp[mask] = (0, 0, 0)

    return disp


def uncert2rgb(disp):
    disp = np.clip(disp, 0, 1.0)
    disp *= 255
    disp = np.round(disp).astype(np.uint8)[..., None]
    disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

    return disp


def depth2rgb(depth, depth_max, depth_min):
    mask = np.logical_or(depth > depth_max, depth < depth_min)
    depth = np.clip(depth, depth_min, depth_max)
    depth = (depth - depth_min) / 4.34 * 0.7 + 0.25
    depth = abs(depth - 1)
    depth *= 255
    depth = np.round(depth).astype(np.uint8)[..., None]
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    depth[mask] = (0, 0, 0)

    return depth


def images_to_video(image_folder, output_video, fps=30):
    images = [
        img
        for img in os.listdir(image_folder)
        if img.endswith((".png", ".jpg", ".jpeg"))
    ]
    images.sort()

    if not images:
        print("No images found in the specified folder.")
        return

    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image_name in tqdm.tqdm(images):
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Warning: Could not read {image_name}. Skipping.")
            continue

        resized_frame = cv2.resize(frame, (width, height))
        video_writer.write(resized_frame)

    video_writer.release()
    print(f"Video saved as {output_video}")


def view_infer_result(
    left, right, pred, disp_gt=None, maxdisp=192, color=(252, 247, 192)
):
    assert left.shape[1] == right.shape[1] == pred.shape[1]
    infra1_infra2 = np.hstack((left, right))
    view_disp = disp2rgb(pred, maxdisp, 1)
    cv2.putText(view_disp, "disp", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    if disp_gt is not None:
        assert disp_gt.shape == left.shape[:2]
        view_gt = disp2rgb(disp_gt, maxdisp, 1)
        cv2.putText(view_gt, "disp gt", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    scend = (
        np.hstack((view_disp, view_gt))
        if disp_gt is not None
        else np.hstack((view_disp, view_disp * 0))
    )
    return np.vstack((infra1_infra2, scend))
