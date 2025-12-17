from copy import deepcopy

from hat.registry import OBJECT_REGISTRY

__all__ = ["TransformBboxToFcosFormat"]


@OBJECT_REGISTRY.register
class TransformBboxToFcosFormat(object):
    def __init__(self, change_label_bool=True):
        self.change_label_bool = change_label_bool

    def __call__(self, data):
        gt_bboxes = data["gt_boxes"][:, :4]
        gt_labels = data["gt_boxes"][:, 4]
        if self.change_label_bool:
            gt_classes = deepcopy(gt_labels)
            gt_classes[gt_labels > 0] -= 1
            gt_classes[gt_labels == 0] = gt_labels.max()
            data["gt_bboxes"] = gt_bboxes
            data["gt_classes"] = gt_classes
        else:
            data["gt_bboxes"] = gt_bboxes
            data["gt_classes"] = gt_bboxes

        return data
