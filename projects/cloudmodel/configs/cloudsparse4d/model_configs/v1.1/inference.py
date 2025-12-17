# 云端推理模型 hdflow model-zoo
# 包含模型结构(model)初始化、前后处理(preprocess、postprocess)的实现
import os
import sys
from typing import List

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../../../../")
)
from config import release_class_id_map as class_id_map
from config import release_score_threshold
from data import get_val_sparse_transforms
from hatbc.message import (
    Attribute,
    BBox3D,
    Frame,
    Instance,
    MessageMeta,
    SyncMessages,
)
from model import get_sparse_model
from torchvision.transforms import Compose

from hat.data.collates.collates import default_collate_v2
from hat.data.transforms.mvt4d_transforms import (
    sync_messages_to_multiview_dict,
)
from hat.registry import build_from_registry

# 模型结构
model = get_sparse_model()


# 模型前处理
def preprocess(data: List[SyncMessages]):
    """
    Preprocessing input data list,
    convert SyncMessage to network input format using val-data-transforms.
    """

    num_batch = len(data)
    batch_data = []

    for i in range(num_batch):
        sync_data = data[i]
        _data = sync_messages_to_multiview_dict(sync_data)
        transforms = Compose(
            build_from_registry(get_val_sparse_transforms(model_setting="x3c"))
        )
        _data = transforms(_data)
        batch_data.append(_data)
    rets = default_collate_v2(batch_data)
    return rets


def postprocess(preds, data):
    """
    Post-processing outputs,
    reformat the output predictions to hatbc.message.Frame format.
    """
    num_batch = len(preds)
    pred_results = []
    for i in range(num_batch):
        pred_mask = preds[i]["scores_3d"] >= release_score_threshold
        pred_bboxes_3d = preds[i]["boxes_3d"][pred_mask].numpy()
        pred_labels_3d = preds[i]["labels_3d"][pred_mask].numpy()
        pred_scores_3d = preds[i]["scores_3d"][pred_mask].numpy()
        num_objs = pred_bboxes_3d.shape[0]

        results = []
        for j in range(num_objs):
            cls = Attribute(
                value=pred_labels_3d[j],
                topic="category",
            )
            bbox3d = BBox3D(
                dim=[
                    pred_bboxes_3d[j, 4],
                    pred_bboxes_3d[j, 5],
                    pred_bboxes_3d[j, 3],
                ],
                loc=pred_bboxes_3d[j, :3],
                yaw=pred_bboxes_3d[j, 6],
                score=pred_scores_3d[j],
                topic="3d_box",
            )
            ins = Instance(
                attributes=[cls],
                bbox3ds=[bbox3d],
                topic="instance",
            )
            results.append(ins)
        ts = int(data["img_metas"]["timestamp"][i].cpu().numpy() * 1e3)
        meta = MessageMeta(timestamp=ts)
        ret = Frame(
            perceptions=results,
            meta=meta,
        )
        pred_results.append(ret)
    return pred_results, data


# 使用Inference进行推理
inference = dict(  # noqa
    type="Inference",
    model=model,
    device=None,
    march="bayes",
    pre_processors=[preprocess],
    post_processors=[postprocess],
    model_convert_pipeline=None,
)
