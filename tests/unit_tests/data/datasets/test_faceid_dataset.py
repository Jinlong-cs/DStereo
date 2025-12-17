import pytest
import torchvision

from hat.data.datasets.faceid_dataset import DeepInsightRecordDataset
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
    "transforms, batch_size", [(None, 1), (transforms, 50)]
)
def test_mxrec_facedataset(transforms, batch_size):
    dataset = DeepInsightRecordDataset(
        rec_path="./tmp_orig_data/face/recognition/mx-record/"
        "face_recognition_test_10cls_122img/"
        "face_recognition_test_10cls_122img.rec",
        idx_path="./tmp_orig_data/face/recognition/mx-record/"
        "face_recognition_test_10cls_122img/"
        "face_recognition_test_10cls_122img.idx",
        transforms=transforms,
        unpack64=True,
    )

    assert len(dataset) == 122

    for ind, data in enumerate(dataset):
        image, target = data["img"], data["labels"]
        print(image.shape, target)
        if ind > 10:
            break
