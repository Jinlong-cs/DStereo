import numpy as np
import pytest
import torch

from hat.core.data_struct.app_struct import (
    DetObject,
    DetObjects,
    build_task_struct,
)
from hat.core.data_struct.base_struct import ClsLabels, DetBoxes2D, Masks
from hat.visualize.multitask.object_render import ObjectRender


class TestObjectRender:
    def _generate_dummy_data(
        self,
        gen_datas=(),
    ):
        # image: [1, H, W, 3]
        h, w = 300, 300
        ret = dict(img=(255 * torch.rand((h, w, 3))).type(torch.uint8))
        # n: num_instance  m: num_class
        n = 3
        m = 4
        assert m <= 4
        # colormaps
        det_colormap = {i: np.array([0, 255, 255]) for i in range(m)}
        ret.update(det_colormap=det_colormap)
        seg_colormap = np.array(
            [
                (220, 20, 60),
                (119, 11, 32),
                (0, 0, 142),
                (0, 0, 230),
            ][:m]
        )
        ret.update(seg_colormap=seg_colormap, ins_colormap=seg_colormap)
        # ins_polygons
        if "ins_polygons" in gen_datas:
            # polygons: [N, 10]
            tmp = torch.rand(n, 4)
            polygons = torch.zeros((n, 10))
            rect_x1 = tmp[:, 0] * h * 0.5
            rect_y1 = tmp[:, 1] * w * 0.5
            rect_x2 = tmp[:, 2] * h + rect_x1
            rect_y2 = tmp[:, 3] * w + rect_y1
            rect_w = rect_x2 - rect_x1
            rect_h = rect_y2 - rect_y1
            tanalpha = (torch.rand(n) - 0.5) * 2 * rect_h / rect_w
            delta_y = rect_w * tanalpha / 2

            polygons[:, 0], polygons[:, 1] = rect_x1, rect_y1
            polygons[:, 2], polygons[:, 3] = rect_x2, rect_y1
            polygons[:, 4], polygons[:, 5] = rect_x2, rect_y2 + delta_y
            polygons[:, 6], polygons[:, 7] = rect_x1, rect_y2 - delta_y
            ret.update(ins_polygons=polygons)

        # seg_masks
        if "seg_masks" in gen_datas:
            ret.update(seg_mask=(torch.rand(h, w) * m).type(torch.int8))
        # ins_set mask
        if "ins_mask":
            ret.update(ins_mask=torch.randn((n, h, w)).bool())
        # ins_rect_boxes
        if "ins_rect_boxes" in gen_datas:
            # box: [N, 4]
            tmp = torch.rand(n, 4)
            rect_x1 = tmp[:, 0] * h * 0.5
            rect_y1 = tmp[:, 1] * w * 0.5
            rect_x2 = tmp[:, 2] * h + rect_x1
            rect_y2 = tmp[:, 3] * w + rect_y1
            ins_rect_boxes = torch.stack(
                [rect_x1, rect_y1, rect_x2, rect_y2], dim=1
            )
            ret.update(ins_rect_boxes=ins_rect_boxes)
        # label: [N]
        if "ins_labels" in gen_datas:
            ins_labels = torch.rand(n).int() * m
            ret.update(ins_labels=ins_labels)
        if "ins_cls_name_mapping" in gen_datas:
            cls_name_mapping = {i: str(i) for i in range(m)}
            ret.update(cls_name_mapping=cls_name_mapping)
        # label: [N]
        if "ins_scores" in gen_datas:
            ins_scores = torch.rand(n).float()
            ret.update(ins_scores=ins_scores)

        return ret

    def test_draw_box2d_objects(self):
        ret = self._generate_dummy_data(
            gen_datas=(
                "ins_rect_boxes",
                "ins_labels",
                "ins_cls_name_mapping",
                "ins_scores",
            )
        )
        # TODO use hatbc message instead
        _, hat_det_struct = build_task_struct(
            "DetObject",
            "DetObjects",
            [("test_draw_box2d_objects", DetBoxes2D)],
            bases=(DetObject, DetObjects),
        )
        image = ret["img"]
        bbox_struct = DetBoxes2D(
            boxes=ret["ins_rect_boxes"],
            scores=ret["ins_scores"],
            cls_idxs=ret["ins_labels"],
            cls_name_mapping=ret["cls_name_mapping"],
        )

        box_res = hat_det_struct(**{"test_draw_box2d_objects": bbox_struct})
        ObjectRender.draw_box2d_objects(
            image,
            box_res,
            colormap=ret["det_colormap"],
        )

    def test_draw_mask2d_objects(self):
        ret = self._generate_dummy_data(
            gen_datas=(
                "ins_mask",
                "ins_labels",
                "ins_scores",
                "ins_cls_name_mapping",
            )
        )
        masks = Masks(ret["ins_mask"])
        attr = ClsLabels(
            cls_idxs=ret["ins_labels"],
            scores=ret["ins_scores"],
            cls_name_mapping=ret["cls_name_mapping"],
        )
        res = dict(
            test_draw_box2d_objects=masks, test_draw_box2d_objects_labels=attr
        )
        # TODO use hatbc message instead
        _, hat_struct = build_task_struct(
            "DetObject",
            "DetObjects",
            [
                ("test_draw_box2d_objects", Masks),
                ("test_draw_box2d_objects_labels", ClsLabels),
            ],
            bases=(DetObject, DetObjects),
        )
        objects = hat_struct(**res)
        ObjectRender.draw_mask2d_objects(
            img=ret["img"], det_objects=objects, colormap=ret["ins_colormap"]
        )

    def test_draw_seg(self):
        ret = self._generate_dummy_data(("seg_masks",))
        ObjectRender.draw_seg(
            img=ret["img"],
            pred_seg=ret["seg_mask"],
            colormap=ret["seg_colormap"],
        )

    def test_draw_polygon2d_objects(self):
        ret = self._generate_dummy_data(
            gen_datas=(
                "ins_polygons",
                "ins_labels",
                "ins_cls_name_mapping",
                "ins_scores",
            )
        )

        image = ret["img"]
        polygon_res = {
            "polygons": ret["ins_polygons"],
            "scores": ret["ins_scores"],
            "cls_idxs": ret["ins_labels"],
            "cls_name": ret["cls_name_mapping"][0],
        }

        ObjectRender.draw_polygon2d_objects(
            image,
            polygon_res,
            colormap=ret["det_colormap"],
        )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
