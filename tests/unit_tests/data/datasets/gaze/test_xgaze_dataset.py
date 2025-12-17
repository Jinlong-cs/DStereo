import os

from hat.data.datasets.gaze.xgaze_dataset import XGazeDataset


def test_dataset():
    x_gaze_path = "/horizon-bucket/MultiMode_2/mm_algorithms_data/gaze/public/eth-xgaze"  # noqa
    if not os.path.isdir(x_gaze_path):
        print("Pass this test.")
        return
    dataset = XGazeDataset(
        dataset_path=x_gaze_path,
        sub_folder="train",
    )
    data = dataset[0]
    assert data["img"].shape == (224, 224, 3)
    assert data["gaze_label"].shape == (2,)
