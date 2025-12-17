import os

import cv2
import numpy as np

from hat.models.task_modules.bev.om_postprocess_utils import sgc_cluster


def test_om_postprocess():
    save_plot_view = False
    bucket_list = ["/horizon-bucket/", "/bucket/input/"]
    data_list = [
        "LS830_20230410_104413_1681094705500_lane",
        "LS830_20230410_102452_1681093534900_lane",
        "LS830_20230410_060429_1681077899100_lane",
        "LS830_20230410_054745_1681076976100_lane",
        "LS830_20230410_080111_1681084873300_roadedge",
        "LS830_20230410_084834_1681087787100_roadedge",
        "LS912_20230410_130400_1681103046699_roadedge",
        "LS912_20230330_141825_1680157173099_roadedge",
    ]
    data_path = "SD_Algorithm/07_perception_bev_static/02_user/fan.zhao/dataset/om/dump_data/model/v11_head_data/"  # noqa
    save_path = "SD_Algorithm/07_perception_bev_static/02_user/fan.zhao/dataset/om/om_post/v2.0/debug/"  # noqa
    for data_name in data_list:
        data = None
        save_view_path = None
        for bucket in bucket_list:
            data_file = os.path.join(bucket, data_path, data_name + ".npy")
            if os.path.exists(data_file):
                data = np.load(data_file, allow_pickle=True)[()]
                save_view_path = os.path.join(bucket, save_path)
                break
        if data is None:
            continue
        pred_cls = data["cls"]
        pred_prob = data["prob"]
        pred_r = data["r"]
        pred_sin = data["sin"]
        pred_cos = data["cos"]
        pred_embedding = data["embedding"]
        cls_thr = data["cls_thr"][0]
        vcs_range = data["vcs_range"].tolist()
        bottom, right, top, left = vcs_range
        min_length = 2.5 if "lane" in data_name else 2.0
        # Notice: if top range > 40, treat it as driving
        if vcs_range[2] > 40:
            min_length *= 2
        cluster_result, sequence_result = sgc_cluster(
            pred_cls,
            pred_prob,
            pred_r,
            pred_sin,
            pred_cos,
            pred_embedding,
            cls_thr,
            vcs_range,
            radius_l=5,
            radius_t=2,
            direction_range=90,
            min_length=min_length,
            inner_thr=0.5,
            cluster_thr=0.85,
            split_channel=True,
        )

        # parse forground instance
        unique_id = np.unique(cluster_result).tolist()
        instance_id_dict = {
            instance_id: i
            for i, instance_id in enumerate(np.unique(unique_id))
        }
        num_pred_lane = len(instance_id_dict)

        lanes = [[] for _ in range(num_pred_lane)]
        out_c, out_h, out_w = pred_cls.shape
        res_h = (top - bottom) / out_h
        res_w = (left - right) / out_w
        unit_res = max(res_h, res_w)
        view_scale = 8
        view_h, view_w = out_h * view_scale, out_w * view_scale
        view_res_h = (top - bottom) / view_h
        view_res_w = (left - right) / view_w
        for h in np.arange(0, out_h):
            for w in np.arange(0, out_w):
                for ch in np.arange(out_c):
                    if not (
                        pred_cls[ch, h, w] > 0
                        and pred_prob[ch, h, w] >= cls_thr
                    ):
                        continue
                    instance_id = instance_id_dict[cluster_result[ch, h, w]]
                    if instance_id <= 0:
                        continue
                    x_origin = top - h * res_h - res_h / 2
                    y_origin = left - w * res_w - res_w / 2
                    r = pred_r[ch, h, w]
                    sin_v = pred_sin[ch, h, w]
                    cos_v = pred_cos[ch, h, w]
                    x_vcs = x_origin - r * cos_v * unit_res
                    y_vcs = y_origin - r * sin_v * unit_res
                    seq_order = sequence_result[ch, h, w]
                    view_x = int((left - y_vcs) / view_res_w)
                    view_y = int((top - x_vcs) / view_res_h)
                    pt = [view_x, view_y, seq_order]
                    lanes[instance_id].append(pt)

        # sequence
        seq_lanes = []
        for lane in lanes:
            seq_lane = sorted(lane, key=lambda x: x[-1])
            seq_lanes.append(seq_lane)

        # plot view
        if not save_plot_view:
            continue
        left_img = np.zeros((view_h, view_w * 2, 3), dtype=np.uint8)
        right_img = np.zeros((view_h, view_w, 3), dtype=np.uint8)
        instance_colors = np.random.rand(256, 3) * 0.75 + 0.25
        instance_colors = (instance_colors * 256).astype(np.uint8).tolist()
        for i in range(len(seq_lanes)):
            lane = seq_lanes[i]
            if len(lane) < 2:
                continue
            instance_color = instance_colors[i % 256]
            for j in range(len(lane) - 1):
                start_pt = tuple(lane[j][:2])
                end_pt = tuple(lane[j + 1][:2])
                left_img = cv2.circle(left_img, end_pt, 2, instance_color, 2)
                right_img = cv2.line(
                    right_img, start_pt, end_pt, instance_color, 2
                )
            # plot start point
            start_pt = tuple(lane[0][:2])
            left_img = cv2.circle(left_img, start_pt, 7, instance_color, 3)
            right_img = cv2.circle(right_img, start_pt, 7, instance_color, 3)
        view_img = np.hstack([left_img, right_img])

        # save
        save_name = "{}/{}.jpg".format(save_view_path, data_name)  # noqa
        cv2.imwrite(save_name, view_img)
