from collections import defaultdict

import numpy as np
import pytest
import torchvision
from torch.utils.data import DataLoader

from hat.data.datasets.facequality_mtl_dataset import FaceQualityDataset
from hat.data.transforms.detection import Normalize

transforms = torchvision.transforms.Compose(
    [
        Normalize(
            mean=(128.0, 128.0, 128.0),
            std=(0.078125, 0.078125, 0.078125),
        ),
    ]
)


@pytest.mark.parametrize(
    "transforms, batch_size, task_name, need_flag",
    [(None, 1, "glass", True), (transforms, 4, ["glass", "hat"], False)],
)
def test_facequality_dataset(transforms, batch_size, task_name, need_flag):
    dataset = FaceQualityDataset(
        rec_path="./tmp_orig_data/face/face_quality/mx-record/mtl/"
        "glass_ir_n32_4ut.rec",
        idx_path="./tmp_orig_data/face/face_quality/mx-record/mtl/"
        "glass_ir_n32_4ut.idx",
        label_path="./tmp_orig_data/face/face_quality/mx-record/mtl/"
        "glass_ir_n32_4ut.label.npy",
        info_path="./tmp_orig_data/face/face_quality/mx-record/mtl/"
        "label_idx_map_v1.0.yaml",
        task_name=task_name,
        need_flag=need_flag,
        transforms=transforms,
    )
    if isinstance(task_name, str):
        task_name = [task_name]
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    assert len(dataset) == 32
    assert len(dataloader) == int(len(dataset) / batch_size)
    label_np = np.load(
        "./tmp_orig_data/face/face_quality/mx-record/mtl/"
        "glass_ir_n32_4ut.label.npy"
    )
    label_glass = label_np[:, 9]
    label_hat = label_np[:, 10]
    gt = {"glass": label_glass, "hat": label_hat}
    # flag
    if need_flag:
        assert np.sum(label_glass - dataset.flag) == 0

    dataset_label = defaultdict(list)
    for _, batch in enumerate(dataloader):
        image, target = batch["img"], batch["gt_face_quality"]
        assert image.shape[0] == batch_size
        assert isinstance(target, dict)
        assert len(target) == len(task_name)
        for k, v in target.items():
            assert k in task_name
            assert v.shape[0] == batch_size
            dataset_label[k].append(v)
    for k, v in dataset_label.items():
        task_label = np.concatenate(v)
        assert np.sum(task_label - gt[k]) == 0
