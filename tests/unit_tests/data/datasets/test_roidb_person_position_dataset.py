from hat.data.datasets.roidb_person_position_dataset import (
    RoidbPersonPositionDataset,
)


def test_roidb_dataset():
    dataset = RoidbPersonPositionDataset(
        data_path="./tmp_orig_data/person_position/T18_dms_smoke_phone_little_data.rec",  # noqa
        anno_path="./tmp_orig_data/person_position/T18_dms_smoke_phone_little_data.pkl",  # noqa
        selected_class_ids=[1],
    )

    for ind, data in enumerate(dataset):
        assert "img" in data
        assert "camera" in data
        assert "gt_position_dms" in data
        assert "gt_position_oms" in data
        assert "gt_classes" in data
        assert "gt_bboxes" in data

        if ind > 5:
            break
