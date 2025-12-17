import json
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from hat.registry import OBJECT_REGISTRY

__all__ = ["PSDPostprocess"]

slot_occupancy_color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}

slot_type_mark_dict = {0: "V", 1: "P", 2: "S"}

point_type_radius_dict = {0: 2, 1: -1}

point_type_color_dict = {0: (0, 255, 255), 1: (0, 0, 0)}


def gen_fuse_result_map(det_results, img):

    # [[junction_locations*8, junction_types*4, junction_visibilities*4,
    # slot_occupancies*1, slot_types*1, junction_orientations*8,
    # slot_orientations*2, slot_score, junction_score*4],...]
    img1, img2 = img.copy(), img.copy()
    det_results = np.array(det_results)
    if len(det_results) == 0:
        return img
    # point_directions = det_results[:, 18:26].reshape(-1, 4, 2)
    junction_locations = det_results[:, 0:8].reshape(-1, 4, 2)
    slot_centers = junction_locations.mean(axis=-2)
    slot_directions = slot_centers + 30 * det_results[:, 26:28]
    # junction_directions = junction_locations + 30 * point_directions
    center_points = junction_locations.mean(1).astype(np.int64)
    for i, junction_location in enumerate(junction_locations):
        cv2.line(
            img1,
            tuple(junction_location[0].astype(np.int64)),
            tuple(junction_location[1].astype(np.int64)),
            (255, 0, 0),
            1,
        )
        cv2.line(
            img1,
            tuple(junction_location[1].astype(np.int64)),
            tuple(junction_location[2].astype(np.int64)),
            (255, 0, 0),
            1,
        )
        cv2.line(
            img1,
            tuple(junction_location[2].astype(np.int64)),
            tuple(junction_location[3].astype(np.int64)),
            (255, 0, 0),
            1,
        )
        cv2.line(
            img1,
            tuple(junction_location[3].astype(np.int64)),
            tuple(junction_location[0].astype(np.int64)),
            (255, 0, 0),
            1,
        )
        cv2.line(
            img1,
            tuple(slot_centers[i].astype(np.int64)),
            tuple(slot_directions[i].astype(np.int64)),
            (255, 0, 100),
            2,
        )

        cv2.fillPoly(
            img2,
            [junction_location.astype(np.int64)],
            slot_occupancy_color_dict[int(det_results[i][16])],
        )
        pass
    for i, center_point in enumerate(center_points):
        cv2.putText(
            img1,
            slot_type_mark_dict[int(det_results[i][17])],
            tuple(center_point),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )
    img1 = cv2.addWeighted(img1, 0.8, img2, 0.2, 0)
    return img1


@OBJECT_REGISTRY.register
class PSDPostprocess(torch.nn.Module):
    """Postprocess for Super PSD algorithm.

        Include two main steps:
        1. Decode globalslot features and localjunction features;
        2. Fuse results from globalslot and localjunction

    Args:
        input_size: Input resolution of model
        downsample_factor_globalslot_feature: Downsample factor of
            global slot feature relative to input size
        downsample_factor_localjunction_feature: Downsample factor
            local junction feature  relative to input size
        threshold_globalslot_feature: Threshold to decode
            slot from global slot feature
        threshold_localjunction_feature: Threshold to decode
            junction from local junction feature
        topk_globalslot_feature: Number of slots
            to decode from global slot feature
        topk_localjunction_feature: Number of junctions
            to decode from local junction feature
        threshold_occupancy_feature: Threshold to classify
            slot occupancy from occupancy feature
        threshold_junction_type_feature: Threshold to classify
            junction type from junction type feature
        threshold_fuse_distance: Threshold of distance to fuse junctions
            decoded from global slot feature and local junction feature
    """

    def __init__(
        self,
        input_size: List[int],
        downsample_factor_globalslot_feature: int,
        downsample_factor_localjunction_feature: int,
        threshold_globalslot_feature: float,
        threshold_localjunction_feature: float,
        topk_globalslot_feature: int = 20,
        topk_localjunction_feature: int = 50,
        threshold_occupancy_feature: float = 0.5,
        threshold_junction_type_feature: float = 0.5,
        threshold_fuse_distance: int = 25,
        global_kernel_size: int = 3,
        local_kernel_size: int = 5,
        draw_flag: bool = False,
        nms_distance_threshold: int = 20,
    ):
        super().__init__()
        self.input_size = input_size
        self.downsample_factor_globalslot_feature = (
            downsample_factor_globalslot_feature  # 16
        )
        self.downsample_factor_localjunction_feature = (
            downsample_factor_localjunction_feature  # 4
        )

        self.globalslot_output_size = [
            i // self.downsample_factor_globalslot_feature
            for i in self.input_size
        ]
        self.localjunction_output_size = [
            i // self.downsample_factor_localjunction_feature
            for i in self.input_size
        ]

        self.threshold_globalslot_feature = threshold_globalslot_feature
        self.threshold_localjunction_feature = threshold_localjunction_feature
        self.topk_globalslot_feature = topk_globalslot_feature
        self.topk_localjunction_feature = topk_localjunction_feature

        self.threshold_occupancy_feature = threshold_occupancy_feature
        self.threshold_junction_type_feature = threshold_junction_type_feature

        self.threshold_fuse_distance = threshold_fuse_distance
        self.global_kernel_size = global_kernel_size
        self.local_kernel_size = local_kernel_size
        self.draw_flag = draw_flag

        self.slot_type_dict = {
            0: "V",
            1: "P",
            2: "O",
            3: "I",
        }  # For visualization
        self.slot_occupancy_color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
        self.nms_distance_threshold = nms_distance_threshold

    def process(self, pred: List[torch.Tensor], meta_data: Dict, batch_idx):
        # copy from dev-superparking branchs
        # original author: @chunyu.bi
        # pred: [global_classification, global_offset, global_slot_occupancy,
        #        global_slot_type, global_slot_orientation,
        #        local_classification, local_offset,
        #        local_sline_angle, local_junction_type]
        if self.draw_flag:
            ori_imgs = meta_data.get("ori_img", None)
            assert ori_imgs is not None
            self.img = ori_imgs[batch_idx]
        slot = self.decode_globalslot_feature(pred[:5])
        junction_dict_list = self.decode_localjunction_feature(pred[5:])
        det_results = self.fuse_globalslot_localjunction(
            slot, junction_dict_list
        )
        det_results = self.slot_nms(det_results, self.nms_distance_threshold)
        if self.draw_flag:
            if "visualized_results" not in meta_data:
                meta_data["visualized_results"] = []
            self.img = gen_fuse_result_map(det_results, self.img)
            meta_data["visualized_results"].append(self.img)
        meta_data["pred_slots"].append(det_results)
        return meta_data

    def post_process(self, pred: List[torch.Tensor], meta_data: Dict):
        # copy from dev-superparking branchs
        # original author: @chunyu.bi
        if self.draw_flag:
            img_path = meta_data.get("img_name", None)
            assert img_path is not None
            self.img = cv2.imread(img_path)
        slot = self.decode_globalslot_feature(pred[:5])
        junction_dict_list = self.decode_localjunction_feature(pred[5:])
        det_results = self.fuse_globalslot_localjunction(
            slot, junction_dict_list
        )
        det_results = self.slot_nms(det_results, self.nms_distance_threshold)
        meta_data.setdefault("pred_slots", det_results)
        if self.draw_flag:
            self.img = gen_fuse_result_map(det_results, self.img)
            meta_data.setdefault("visualized_results", self.img)
        return meta_data

    def slot_nms(self, det_results, nms_distance_threshold):
        """Use to erase repeated detection results.

        Args:
            det_results: detection results after fusing global head \
                and local head.
            iou_threshold: nms iou threshold.
        """
        # det_results: [junction_locations*8, junction_types*4,
        # junction_visibilities*4, slot_occupancies*1,
        # slot_types*1, junction_orientations*8,
        # slot_orientations*2, slot_score, junction_score*4]
        if len(det_results) == 0:
            return det_results
        det_results = np.array(det_results)
        scores = det_results[:, 28]
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            idx = order[0]
            keep.append(det_results[idx].tolist())
            bbox1 = det_results[idx, :8].reshape(4, 2)
            bbox1_center = np.mean(bbox1, axis=0)
            rest_bbox = det_results[order[1:], :8].reshape(-1, 4, 2)
            rest_bbox_center = np.mean(rest_bbox, axis=1)
            distance = np.sqrt(
                np.sum((bbox1_center - rest_bbox_center) ** 2, axis=-1)
            )
            index = np.where(distance > nms_distance_threshold)[0]
            order = order[index + 1]
        return keep

    def decode_globalslot_feature(
        self, globalslot_feature
    ) -> Dict[str, List[torch.Tensor]]:
        """Decode slot information from global slot feature.

        Args:
            globalslot_feature: tensor of globalslot_feature from the model
        Returns:
            A dict mapping keys to the corresponding list
        """
        # Postprocess feature
        # objectness score of slot center
        classification_slot = torch.sigmoid(globalslot_feature[0]).squeeze()
        # (x,y) offset of four junctions relative to slot tile center
        offset_junctions = (
            globalslot_feature[1].squeeze()
            * self.downsample_factor_globalslot_feature
        )
        # occupancy of the slot by vehicles
        occupancy = torch.sigmoid(globalslot_feature[2]).squeeze()
        if globalslot_feature[3].shape[0] != 1:
            type_slot = torch.argmax(
                globalslot_feature[3].squeeze(), dim=0
            )  # Moved into BPU
        else:
            type_slot = globalslot_feature[3].squeeze()
        # sin(theta), cos(theta) of slot
        orientation = F.normalize(globalslot_feature[4].squeeze(), p=2, dim=0)

        # Decode result from feature
        slot_junctions_locations = []
        slot_scores = []
        slot_occupancies = []
        slot_types = []
        slot_orientations = []
        center_indices, center_values = self.decode_object(
            classification_slot,
            self.threshold_globalslot_feature,
            self.topk_globalslot_feature,
            self.global_kernel_size,
        )
        tile_center = self.generate_tile_centers(
            self.downsample_factor_globalslot_feature,
            self.globalslot_output_size,
        )
        tile_center = tile_center.to(offset_junctions.device)
        for i, center_index in enumerate(center_indices):
            offset_junctions_i = offset_junctions[
                :, center_index[0], center_index[1]
            ].reshape(4, 2)
            tile_center_i = tile_center[:, center_index[0], center_index[1]]
            slot_junctions_locations_i = (
                self.downsample_factor_globalslot_feature
                * tile_center_i[None, :]
                + offset_junctions_i
            )

            occupancy_i = occupancy[center_index[0], center_index[1]]
            type_slot_i = type_slot[center_index[0], center_index[1]]
            orientation_i = orientation[:, center_index[0], center_index[1]]

            if self.draw_flag:
                assert self.img is not None
                slot_junctions_locations_i_plot = slot_junctions_locations_i
                point_locations = (
                    slot_junctions_locations_i_plot.int().tolist()
                )
                for _, point_location in enumerate(point_locations):
                    cv2.circle(
                        self.img, tuple(point_location), 2, (0, 0, 255), 2
                    )
                    # cv2.putText(self.img, str(idx),
                    # tuple(point_location),
                    # cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

            slot_junctions_locations.append(slot_junctions_locations_i)
            slot_occupancies.append(occupancy_i)
            slot_types.append(type_slot_i)
            slot_orientations.append(orientation_i)
            slot_scores.append(center_values[i])

        return {
            "slot_junctions_locations_list": slot_junctions_locations,
            "slot_occupancy_list": slot_occupancies,
            "slot_type_list": slot_types,
            "slot_orientation_list": slot_orientations,
            "slot_score_list": slot_scores,
        }

    def decode_localjunction_feature(
        self, localjunction_feature: torch.Tensor
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Decode junction information from local junction feature.

        Args:
            localjunction_feature: tensor of localjunction_feature \
                from the model
        Returns:
            Dicts of junction0, junction1, junction2, junction3
            mapping keys to the corresponding list
        """
        # Postprocess feature
        junction_dict_list = []
        tile_center = self.generate_tile_centers(
            self.downsample_factor_globalslot_feature,
            self.localjunction_output_size,
        )
        tile_center = tile_center.to(localjunction_feature[0].device)
        for idx in range(4):

            classification = torch.sigmoid(
                localjunction_feature[0][idx]
            ).squeeze()
            offset = localjunction_feature[1][2 * idx : 2 * idx + 2].squeeze()
            sline_angle = F.normalize(
                localjunction_feature[2][2 * idx : 2 * idx + 2].squeeze(),
                p=2,
                dim=0,
            )
            type_junction = torch.sigmoid(
                localjunction_feature[3][idx]
            ).squeeze()
            j_indices, j_values = self.decode_object(
                classification,
                self.threshold_localjunction_feature,
                self.topk_localjunction_feature,
                self.local_kernel_size,
            )
            locations, sline_angles, junction_types, scores = [], [], [], []
            for index, junction_index in enumerate(j_indices):
                offset_junction_i = offset[
                    :, junction_index[0], junction_index[1]
                ]
                tile_center_i = tile_center[
                    :, junction_index[0], junction_index[1]
                ]
                junction_location_i = tile_center_i + offset_junction_i
                junction_location_i *= (
                    self.downsample_factor_localjunction_feature
                )  # Newly added
                sline_anlge_junction_i = sline_angle[
                    :, junction_index[0], junction_index[1]
                ]
                type_junction_i = type_junction[
                    junction_index[0], junction_index[1]
                ]
                locations.append(junction_location_i)
                sline_angles.append(sline_anlge_junction_i)
                junction_types.append(type_junction_i)
                scores.append(j_values[index])

                if self.draw_flag:
                    assert self.img is not None
                    junction_location_i_plot = junction_location_i.int()
                    junction_location_direction = (
                        junction_location_i_plot + 50 * sline_anlge_junction_i
                    )
                    junction_location_direction = (
                        junction_location_direction.int()
                    )
                    junction_location_i_plot = (
                        junction_location_i_plot.tolist()
                    )
                    junction_location_direction = (
                        junction_location_direction.tolist()
                    )
                    cv2.circle(
                        self.img,
                        tuple(junction_location_i_plot),
                        2,
                        (0, 255, 0),
                        2,
                    )
                    # cv2.putText(self.img, str(idx),
                    # tuple(junction_location_i_plot),
                    # cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))

            junction_dict = {
                f"j{idx}_location_list": locations,
                f"j{idx}_sline_angle_list": sline_angles,
                f"j{idx}_type_list": junction_types,
                f"j{idx}_score_list": scores,
            }
            junction_dict_list.append(junction_dict)
        return junction_dict_list

    def fuse_globalslot_localjunction(self, slot, junction_dict_list):
        """Fuse results from globalslot and localjunction.

        Args:
            slot: results of slot decoded from globalslot, i.e., \
            {"slot_junctions_locations_list": slot_junctions_locations, \
            "slot_occupancy_list": slot_occupancies,"slot_type_list": \
            slot_types, "slot_orientation_list": slot_orientations}
            j0: results of junction0 decoded from localjunction, i.e., \
            {"j0_location_list":j0_locations, "j0_sline_angle_list": \
            j0_sline_angles, "j0_type_list":j0_types}
            j1: results of junction1 decoded from localjunction, i.e., \
            {"j1_location_list":j1_locations, "j1_sline_angle_list": \
            j1_sline_angles, "j1_type_list":j1_types}
            j2: results of junction2 decoded from localjunction, i.e., \
            {"j2_location_list":j2_locations, "j2_sline_angle_list": \
            j2_sline_angles, "j2_type_list":j2_types}
            j3: results of junction3 decoded from localjunction, i.e., \
            {"j3_location_list":j3_locations, "j3_sline_angle_list": \
            j3_sline_angles, "j3_type_list":j3_types}
        Returns:
            det_results: the output slots locations and attributions, \
            shape (N, 28), N-represents slot numbers, 28 represents \
            [x0,y0,x1,y1,x2,y2,x3,y3, type_j0, type_j1, type_j2, \
            type_j3,visibility_j0, visibility_j1, visibility_j2, \
            visibility_j3, occupancy of slot, type of slot, \
            sin_j0, cos_j0, sin_j1, cos_j1, sin_j2, \
            cos_j2, sin_j3, cos_j3, sin_slot, cos_slot]
        """
        det_results = []
        slot_scores = slot["slot_score_list"]
        sorted_score_idx_list = sorted(
            range(len(slot_scores)), key=lambda i: slot_scores[i], reverse=True
        )
        chosen_point_list = [{}, {}, {}, {}]
        for idx in sorted_score_idx_list:
            slot_junctions = slot["slot_junctions_locations_list"][idx]
            det_result = []
            total_score = 0
            slot_occupancy = (
                1
                if slot["slot_occupancy_list"][idx]
                > self.threshold_occupancy_feature
                else 0
            )
            slot_type = slot["slot_type_list"][idx]
            slot_orientation = slot["slot_orientation_list"][idx]
            slot_score = slot["slot_score_list"][idx]
            visibility, type_junction, orientation, score = (
                0,
                0,
                torch.tensor([0, 0]),
                torch.tensor(0.1),
            )
            (
                coord_list,
                type_junction_list,
                visibility_list,
                orientation_list,
                score_list,
            ) = ([], [], [], [], [])
            for index in range(4):
                if (
                    len(junction_dict_list[index][f"j{index}_location_list"])
                    > 0
                ):
                    locations_tensor = torch.cat(
                        [
                            junction.unsqueeze(0)
                            for junction in junction_dict_list[index][
                                f"j{index}_location_list"
                            ]
                        ],
                        dim=0,
                    )
                    # distance of j0 decoded from globalslot and localjunction
                    slot_dist = torch.sqrt(
                        torch.sum(
                            (slot_junctions[index][None, :] - locations_tensor)
                            ** 2,
                            dim=1,
                        )
                    )
                    min_slot_dist, selected_j = torch.min(slot_dist, 0)
                    if min_slot_dist < self.threshold_fuse_distance:
                        flag = chosen_point_list[index].get(
                            selected_j.item(), False
                        )
                        if not flag:
                            chosen_point_list[index][selected_j.item()] = True
                            total_score += 1
                            # Substitute junction 0 locations
                            # by the locations decoded from localjunction
                            slot_junctions[index] = locations_tensor[
                                selected_j
                            ]
                            visibility = 1
                            score = junction_dict_list[index][
                                f"j{index}_score_list"
                            ][selected_j]
                            type_junction = (
                                1
                                if junction_dict_list[index][
                                    f"j{index}_type_list"
                                ][selected_j]
                                > self.threshold_junction_type_feature
                                else 0
                            )
                            orientation = junction_dict_list[index][
                                f"j{index}_sline_angle_list"
                            ][selected_j]
                coord_list.append(slot_junctions[index][0].item())
                coord_list.append(slot_junctions[index][1].item())
                type_junction_list.append(type_junction)
                visibility_list.append(visibility)
                orientation_list.append(orientation[0].item())
                orientation_list.append(orientation[1].item())
                score_list.append(score.item())
            det_result.extend(coord_list)
            det_result.extend(type_junction_list)
            det_result.extend(visibility_list)
            det_result.append(slot_occupancy)
            det_result.append(slot_type.item())
            det_result.extend(orientation_list)
            det_result.extend(
                [slot_orientation[0].item(), slot_orientation[1].item()]
            )
            det_result.append(slot_score.item())
            det_result.extend(score_list)
            if total_score > 1:
                det_results.append(det_result)
                # [junction_locations*8, junction_types*4,
                # junction_visibilities*4, slot_occupancies*1,
                # slot_types*1, junction_orientations*8,
                # slot_orientations*2, slot_score, junction_score*4]
        return det_results

    def decode_object(
        self,
        feature: torch.Tensor,
        threshold: float,
        topk_number: int,
        kernel_size: int,
    ):
        """Decode object from the feature of objectness score."""
        maximum = F.max_pool2d(
            feature.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        maximum = torch.eq(feature, maximum)
        feature = feature * maximum
        feature = feature.squeeze()
        # feature = torch.where(feature>threshold,feature,0.)
        # in pytorch version >=1.8.0
        feature[feature < threshold] = 0
        H, W = feature.size()
        feature = feature.view(-1)
        values, indices = feature.topk(topk_number, dim=0)
        indices = indices[values > 0]
        values = values[values > 0]
        # [y1, y2, y3,...., x1, x2, x3,...]
        indices = torch.cat(
            [torch.div(indices, W).long(), torch.remainder(indices, W).long()]
        )
        indices = indices.view(2, -1)  # [[y1, y2, y3, ...], [x1, x2, x3,...]]
        indices = indices.transpose(1, 0)  # [[y1, x1], [y2, x2], [y3, x3],...]
        return indices, values

    def generate_tile_centers(
        self, downsample_factor: int, output_size: List[int]
    ) -> torch.Tensor:
        """Generate tile centers."""
        tile_center_x = [n for n in range(output_size[0])]  # noqa
        tile_center_y = [n for n in range(output_size[1])]  # noqa
        tile_center = torch.empty(
            size=[
                2,
            ]
            + output_size
        )
        tile_center[1, :, :] = torch.tensor(tile_center_x)[:, None]
        tile_center[0, :, :] = torch.tensor(tile_center_y)
        return tile_center

    def tensor2list(self, data):
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                result[key] = self.tensor2list(value)
            return result
        elif isinstance(data, (list, tuple)):
            return [self.tensor2list(d) for d in data]
        elif isinstance(data, (int, float, str)):
            return data
        elif isinstance(data, torch.Tensor):
            return data.tolist()

    def __call__(
        self,
        preds: tuple,
        label=None,
    ) -> Dict:
        """Get target from label.

        Args:
            preds (tuple):
                global_classification:     torch.randn((1, 1, 48, 32)),
                global_offset:             torch.randn((1, 8, 48, 32)),
                global_occupancy:          torch.randn((1, 1, 48, 32)),
                global_slot_type:          torch.randn((1, 1, 48, 32)),
                global_direction:          torch.randn((1, 2, 48, 32)),
                local_classification:     torch.randn((1, 4, 192, 128)),
                local_offset:             torch.randn((1, 8, 192, 128)),
                local_sline_angle:        torch.randn((1, 8, 192, 128)),
                local_point_type:         torch.randn((1, 4, 192, 128)),
            label (Dict, optional): _description_. Defaults to None.

        Returns:
            Dict: _description_
        """
        preds_global, preds_local, preds_slots = [], [], []
        for pred in zip(
            preds[0],
            preds[1],
            preds[2],
            preds[3],
            preds[4],
            preds[5],
            preds[6],
            preds[7],
            preds[8],
        ):
            pred = list(pred)
            pred_global = self.decode_globalslot_feature(pred[:5])
            pred_local = self.decode_localjunction_feature(pred[5:])
            pred_slots = self.fuse_globalslot_localjunction(
                pred_global, pred_local
            )
            pred_slots = self.slot_nms(pred_slots, self.nms_distance_threshold)
            preds_global.append(pred_global)
            preds_local.append(pred_local)
            preds_slots.append(pred_slots)
        # preds_slots_=self.tensor2list(preds_slots)
        preds_label = {
            "preds_global": self.tensor2list(preds_global),
            "preds_local": self.tensor2list(preds_local),
            "preds_slots": self.tensor2list(preds_slots),
        }
        return {
            "preds_label": json.dumps(preds_label),
        }
