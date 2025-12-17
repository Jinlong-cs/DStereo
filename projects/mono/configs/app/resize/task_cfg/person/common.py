from functools import partial

from ...common import loss_weights
from ..vdvru_common import get_model_mtf3d, roi_feat_extractor, roi_head

classnames = ["person"]
object_type = "_".join(classnames)
num_classes = 1

roi_head["node_name"] = f"{object_type}_roi_share_head"
roi_feat_extractor["node_name"] = f"{object_type}_roi_feat_extractor"
disentangled_corner3D = False

model_common_keys = dict(
    object_type=object_type,
    loss_2d_2dBatch_weight=loss_weights[f"{object_type}_detection"],
    lr_factor=1.0,
    reg_2d_factor=1.0,
    disentangled_corner3D=disentangled_corner3D,
)

get_model = partial(
    get_model_mtf3d,
    **model_common_keys,
)
