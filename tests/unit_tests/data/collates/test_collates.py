import numpy as np
import pytest
import torch

from hat.data.collates.collates import (
    CocktailCollate,
    collate_2d,
    collate_2d_cat,
    collate_2d_pad,
    collate_2d_replace_empty,
    collate_2d_with_diff_im_hw,
    collate_argoverse,
    collate_disp_cat,
    collate_e2e_dynamic,
    collate_gaze_seq,
    collate_mot_seq,
    collate_psd,
    collate_seq_with_diff_im_hw,
)


def test_collate_mot_seq():
    x11 = dict(
        img=np.random.random((3, 100, 120)),
    )
    x12 = dict(
        img=np.random.random((3, 200, 80)),
    )
    frame_seq1 = [x11, x12]
    data1 = {
        "frame_data_list": frame_seq1,
        "frame_length": 2,
    }

    x21 = dict(
        img=np.random.random((3, 100, 120)),
    )
    x22 = dict(
        img=np.random.random((3, 200, 80)),
    )
    frame_seq2 = [x21, x22]
    data2 = {
        "frame_data_list": frame_seq2,
        "frame_length": 2,
    }
    seq_batch = [data1, data2]
    batch_collated = collate_mot_seq(seq_batch)
    assert isinstance(batch_collated, dict)


def test_collate_2d():
    x = torch.randn((1, 3, 100, 100))
    batch = [x, x]
    batch_collated = collate_2d(batch)
    assert isinstance(batch_collated, torch.Tensor)

    x = dict(
        image_file="test.jpg",
        classes=[0, 1, 0],
        bboxes=[
            torch.tensor([0, 0, 10, 10]),
            torch.tensor([0, 0, 20, 20]),
        ],
    )
    batch = [x, x]
    batch_collated = collate_2d(batch)
    assert isinstance(batch_collated, dict)
    assert "image_file" in batch_collated
    assert "classes" in batch_collated
    assert "bboxes" in batch_collated


def test_collate_psd():
    x = torch.randn((1, 3, 896, 896))
    batch = [x, x]
    batch_collated = collate_2d(batch)
    assert isinstance(batch_collated, torch.Tensor)
    x = dict(
        img=torch.randn((1, 3, 896, 896)),
        label=[[], []],
        ori_img=torch.randn((1, 3, 896, 896)),
        img_name="test.jpg",
    )
    batch = [x, x]
    batch_collated = collate_psd(batch)
    assert isinstance(batch_collated, dict)
    assert "img" in batch_collated
    assert "label" in batch_collated
    assert "ori_img" in batch_collated
    assert "img_name" in batch_collated


@pytest.mark.skipif(True, reason="skip test collate_bev, it need refactor")
def test_collate_3d():
    pass


@torch.no_grad()
def test_cocktail_collate():
    cocktial_collate = CocktailCollate(ignore_id=-1, batch_first=True)
    images = [
        torch.randn((10, 3, 96, 96)),
        torch.randn((15, 3, 96, 96)),
        torch.randn((8, 3, 96, 96)),
    ]
    audio = [
        torch.randn((40, 80)),
        torch.randn((60, 80)),
        torch.randn((32, 80)),
    ]
    label = [
        torch.tensor([1, 2, 3, 4]),
        torch.tensor([1, 2, 3, 4, 5, 6]),
        torch.tensor([1, 2, 3]),
    ]
    images_lens = [10, 15, 8]
    audio_lens = [40, 60, 32]
    tokens = [
        ["a", "b", "c", "d"],
        ["a", "b", "c", "d", "e", "f"],
        ["a", "b", "c"],
    ]
    batch = [
        {
            "images": i,
            "audio": a,
            "label": l,
            "tokens": t,
            "images_lens": il,
            "audio_lens": al,
        }
        for i, a, l, il, al, t in zip(
            images, audio, label, images_lens, audio_lens, tokens
        )
    ]
    batch = cocktial_collate(batch)
    assert batch["images"].shape == torch.Size([3, 15, 3, 96, 96])
    assert batch["audio"].shape == torch.Size([3, 60, 80])
    assert batch["label"].shape == torch.Size([3, 6])
    assert batch["images_lens"].shape == torch.Size([3])
    assert batch["audio_lens"].shape == torch.Size([3])
    target_images = torch.stack(
        [
            torch.cat((images[0], torch.zeros((5, 3, 96, 96))), dim=0),
            images[1],
            torch.cat((images[2], torch.zeros((7, 3, 96, 96))), dim=0),
        ],
        dim=0,
    )
    assert all(batch["images"].flatten() == target_images.flatten())
    target_audio = torch.stack(
        [
            torch.cat((audio[0], torch.zeros((20, 80))), dim=0),
            audio[1],
            torch.cat((audio[2], torch.zeros((28, 80))), dim=0),
        ],
        dim=0,
    )
    assert all(batch["audio"].flatten() == target_audio.flatten())
    target_lebel = torch.tensor(
        [
            [1, 2, 3, 4, -1, -1],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, -1, -1, -1],
        ]
    )
    assert all(target_lebel.flatten() == batch["label"].flatten())
    assert all(
        all(tt == bt for tt, bt in zip(target_token, batch_token))
        for target_token, batch_token in zip(tokens, batch["tokens"])
    )
    target_images_lens = torch.tensor(images_lens)
    assert all(batch["images_lens"] == target_images_lens)
    target_audio_lens = torch.tensor(audio_lens)
    assert all(batch["audio_lens"] == target_audio_lens)


def test_collate_2d_with_diff_im_shape():
    x = torch.randn((1, 3, 100, 100))
    batch = [x, x]
    batch_collated = collate_2d_with_diff_im_hw(batch)
    assert isinstance(batch_collated, torch.Tensor)

    x1 = dict(
        img=np.random.random((3, 100, 120)),
    )
    x2 = dict(
        img=np.random.random((3, 200, 80)),
    )
    batch = [x1, x2]
    batch_collated = collate_2d_with_diff_im_hw(batch)
    assert isinstance(batch_collated, dict)
    assert "img" in batch_collated


def test_collate_2d_replace_empty():
    x1 = dict(
        img_name="x1.jpg",
        img=torch.randn((3, 100, 100)),
        gt_classes=torch.tensor([0, 1]),
        gt_bboxes=[
            torch.tensor([0, 0, 10, 10]),
            torch.tensor([0, 0, 20, 20]),
        ],
    )
    x2 = dict(
        img_name="x2.jpg",
        img=torch.randn((3, 100, 100)),
        gt_classes=torch.tensor([-1, -1]),
        gt_bboxes=[
            torch.tensor([0, 0, 10, 10]),
            torch.tensor([0, 0, 20, 20]),
        ],
    )
    batch = [x1, x2]
    batch_collated = collate_2d_replace_empty(batch, prob=1.0)
    assert isinstance(batch_collated, dict)
    assert batch_collated["img"].ndim == 4
    assert "img_name" in batch_collated
    assert "img" in batch_collated
    assert "gt_classes" in batch_collated
    assert "gt_bboxes" in batch_collated
    assert "x2.jpg" not in batch_collated["img_name"]


def test_collate_seq_with_diff_im_shape():
    x11 = dict(
        img=np.random.random((3, 100, 120)),
    )
    x12 = dict(
        img=np.random.random((3, 200, 80)),
    )
    frame_seq1 = [x11, x12]
    data1 = {
        "frame_data_list": frame_seq1,
        "frame_length": 2,
    }

    x21 = dict(
        img=np.random.random((3, 100, 120)),
    )
    x22 = dict(
        img=np.random.random((3, 200, 80)),
    )
    frame_seq2 = [x21, x22]
    data2 = {
        "frame_data_list": frame_seq2,
        "frame_length": 2,
    }

    seq_batch = [data1, data2]
    batch_collated = collate_seq_with_diff_im_hw(seq_batch)
    assert isinstance(batch_collated, dict)
    assert "img" in batch_collated and len(batch_collated["img"]) == 4
    assert (
        "num_seq" in batch_collated and batch_collated["num_seq"].item() == 2
    )
    assert "seq_len" in batch_collated and batch_collated["seq_len"][0] == 2


def test_collate_2d_pad():
    x = torch.randn((3, 100, 100))
    y = torch.randn((3, 120, 240))
    batch = [x, y]
    batch_collated = collate_2d_pad(batch)
    assert isinstance(batch_collated, torch.Tensor)

    x1 = dict(
        img_name="x1.jpg",
        img=torch.randn((3, 100, 120)),
    )
    x2 = dict(
        img_name="x2.jpg",
        img=torch.randn((3, 200, 80)),
    )
    batch = [x1, x2]
    batch_collated = collate_2d_pad(batch)
    assert isinstance(batch_collated, dict)
    assert "img_name" in batch_collated
    assert "img" in batch_collated


def test_collate_2d_cat():
    x1 = dict(
        img_name="x1.jpg",
        img=torch.randn((1, 3, 128, 128)),
        vehicle_detection=torch.randn((2, 5)),
        person_detection=torch.randn((1, 5)),
        num_boxes={"vehicle_detection": 2, "person_detection": 1},
    )
    x2 = dict(
        img_name="x2.jpg",
        img=torch.randn((2, 3, 128, 128)),
        vehicle_detection=torch.randn((1, 5)),
        person_detection=torch.randn((1, 5)),
        num_boxes={"vehicle_detection": 1, "person_detection": 1},
    )
    batch = [x1, x2]
    batch_collated = collate_2d_cat(batch)
    assert isinstance(batch_collated, dict)
    assert "img" in batch_collated
    assert "vehicle_detection" in batch_collated
    assert "person_detection" in batch_collated
    assert batch_collated["img"].shape[0] == 3
    assert batch_collated["vehicle_detection"].shape[0] == 3
    assert batch_collated["person_detection"].shape[0] == 2


def test_collate_argoverse():

    data = {
        "traj_feat": torch.randn((1, 9, 19, 32)),
        "lane_feat": torch.randn((1, 11, 9, 64)),
        "instance_mask": torch.randn((1, 1, 96)),
        "goals_2d": torch.randn((1, 2, 2048)),
        "goals_2d_mask": torch.randn((1, 2048)),
        "traj_labels": torch.randn((1, 30, 2)),
        "goals_2d_labels": torch.ones((1)).long(),
        "end_points": torch.randn((1, 1, 2)),
    }

    batch = [data] * 2
    batch = collate_argoverse(batch)
    assert "traj_feat" in batch
    assert "lane_feat" in batch
    assert "instance_mask" in batch
    assert "goals_2d" in batch
    assert "goals_2d_mask" in batch
    assert "traj_labels" in batch
    assert "goals_2d_labels" in batch
    assert "end_points" in batch


def test_collate_disp_cat():

    x1 = dict(
        img=torch.randn((2, 3, 128, 128)),
    )
    x2 = dict(
        img=torch.randn((2, 3, 128, 128)),
    )
    batch = [x1, x2]
    batch_collated = collate_disp_cat(batch)
    assert isinstance(batch_collated, dict)
    assert "img" in batch_collated
    assert torch.all(batch_collated["img"][0] == x1["img"][0])
    assert torch.all(batch_collated["img"][1] == x2["img"][0])
    assert torch.all(batch_collated["img"][2] == x1["img"][1])
    assert torch.all(batch_collated["img"][3] == x2["img"][1])


def test_collate_gaze_seq():

    x1 = dict(
        img=torch.randn((3, 128, 128)),
    )
    x2 = dict(
        img=torch.randn((3, 128, 128)),
    )
    batch = [[x1, x2], [x1, x2]]
    batch_collated = collate_gaze_seq(batch)
    assert isinstance(batch_collated, dict)
    assert batch_collated["img"].ndim == 4
    assert "img" in batch_collated


def test_collate_e2e_dynamic():
    data = dict(  # noqa [C408]
        img=[torch.randn((2, 3, 512, 960))],
        side_img=[torch.randn((10, 3, 640, 1024))],
        timestamp=torch.randn((1,)),
        motr_targets={
            "bevtrack_ing": [
                {
                    "obj_idxes": torch.zeros(1, 100, dtype=torch.int64),
                    "labels": torch.zeros(1, 100, dtype=torch.int64),
                    "yaws": torch.zeros(1, 100, 2, dtype=torch.float32),
                    "bev_loc_z": torch.zeros(1, 100, 2, dtype=torch.float32),
                    "heights": torch.zeros(1, 100, dtype=torch.float32),
                    "scores": torch.zeros(1, 100, dtype=torch.float32),
                }
            ]
            * 2,
            "trajectory_pred": [
                {
                    "ego_raw": torch.zeros(12, 3, dtype=torch.float32),
                    "ego_hisArrs": torch.zeros(12, 3, dtype=torch.float32),
                    "targets_raw": torch.zeros(60, 3, dtype=torch.float32),
                }
            ]
            * 2,
        },
        veh_gt={
            "gt_bev_3d": {
                "bev3d_hm": torch.zeros(2, 1, 224, 256, dtype=torch.float32),
                "bev3d_ignore_mask": torch.zeros(
                    2, 1, 224, 256, dtype=torch.float32
                ),
                "bev3d_weight_hm": torch.zeros(
                    2, 1, 224, 256, dtype=torch.float32
                ),
            }
        },
        meta_info={
            "homography": torch.randn((2, 6, 3, 3)),
            "homo_offset": torch.randn((12, 256, 256, 2)),
            "T_vcs2cam": [torch.randn((1, 4, 4))],
            "persp_view_scale": 0.25,
            "calib_path": "/path/to/calib",
        },
        view="front_side",
        pack_dir="xxxxxx/xxxxx",
        img_paths=["aa.png", "bb.png", "cc.png"],
        pil_imgs=[["test.png"] * 11, ["test.png"] * 11],
        pack_names="abcd",
        odo_info=torch.randn((2, 12, 3)),
    )

    batch = [data, data, data]  # batch size =3
    batch_collated = collate_e2e_dynamic(batch)

    assert isinstance(batch_collated, dict)
    assert "img" in batch_collated
    assert batch_collated["img"][0].shape[0] == 6
    assert len(batch_collated["pack_dir"]) == 3
    assert len(batch_collated["motr_targets"]) == 3


if __name__ == "__main__":
    pytest.main(["-s", __file__])
