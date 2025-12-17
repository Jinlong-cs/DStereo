import os

from hat.data.datasets.roidb_act_dataset import RoidbActDataset

PATH_DATA = (
    "./tmp_orig_data/action/gesture"
    if not os.path.isdir("/horizon-bucket/HDLTAlgorithm/")
    else "/horizon-bucket/HDLTAlgorithm/data/orig_data/action/gesture"
)


def test_roidb_dataset():
    path_data = "{}/hgr_dac_dag/".format(PATH_DATA)
    is_read_rec = True
    dataset = RoidbActDataset(
        rec_path=os.path.join(
            path_data,
            "img_rec/CD569_hand_IFRC_IFARC_220121_220126_testset_dynamic.rec",
        ),
        roidb_path=os.path.join(
            path_data,
            "roidbs/CD569_hand_IFRC_IFARC_220121_220126_testset_det_kps_v2.0_ldmk2.5dv3.0.3_roidb.pkl",  # noqa: E501
        ),
        roidb_seq_path=os.path.join(
            path_data,
            "roidbs/CD569_hand_IFRC_IFARC_220121_220126_testset_act_v2.0_ldmk2.5dv3.0.3_roidb.pkl",  # noqa: E501
        ),
        is_read_rec=is_read_rec,
    )
    assert len(dataset) == dataset.flag.shape[0]

    for ind, data in enumerate(dataset):
        assert "video_roidb" in data
        assert "roidbs" in data

        if is_read_rec:
            assert "rec_packer" in data
        if ind > 10:
            break
