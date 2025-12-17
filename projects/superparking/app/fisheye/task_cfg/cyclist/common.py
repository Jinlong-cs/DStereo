import copy
from functools import partial

from ..vdvru_common import _roi_feat_extractor, _roi_head, get_model_mtf3d

classnames = ["cyclist"]
object_type = "_".join(classnames)
num_classes = 1

roi_head = copy.deepcopy(_roi_head)
roi_head["node_name"] = f"{object_type}_roi_share_head"
roi_feat_extractor = copy.deepcopy(_roi_feat_extractor)
roi_feat_extractor["node_name"] = f"{object_type}_roi_feat_extractor"
disentangled_corner3D = False

get_model = partial(
    get_model_mtf3d,
    object_type=object_type,
    loss_2d_2dBatch_weight=10.0,
    lr_factor=1.0,
    reg_2d_factor=1.0,
    disentangled_corner3D=disentangled_corner3D,
)
