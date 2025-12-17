import pytest

from hat.data.datasets.roidb_track_dataset import RoidbTrackDataset


@pytest.mark.parametrize(
    [
        "epoch",
    ],
    [
        pytest.param(0),
        pytest.param(1),
    ],
)
@pytest.mark.parametrize(
    [
        "ign_heavy_occlusion",
    ],
    [
        pytest.param(False),
        pytest.param(True),
    ],
)
def test_roidb_track_dataset(epoch, ign_heavy_occlusion):
    epoch_seq_length_map = {
        0: 2,
        1: 4,
    }
    dataset = RoidbTrackDataset(
        data_path="./tmp_data/halo_detection/track/test_datasets_img_rec.rec",
        anno_path="./tmp_data/halo_detection/track/test_datasets_roidb.pkl",
        anno_seq_path="./tmp_data/halo_detection/track/test_datasets_act_roidb.pkl",  # noqa: E501
        selected_class_ids=[4],
        max_sample_interval=6,
        ign_heavy_occlusion=ign_heavy_occlusion,
        epoch_seq_length_map=epoch_seq_length_map,
        init_seq_length=2,
    )

    dataset.set_epoch(epoch)
    for ind, data in enumerate(dataset):
        assert len(data["frame_data_list"]) == epoch_seq_length_map[epoch]
        assert "gt_ids" in data["frame_data_list"][0]

        if ind > 10:
            break


if __name__ == "__main__":
    pytest.main(["-s", __file__])
