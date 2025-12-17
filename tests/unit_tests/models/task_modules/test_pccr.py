import torch

from hat.models.task_modules.PCCR_gaze.pccr import PCCRProcess


def gen_fake_data():
    pbn = 6
    imgn = 8
    data = {
        "R": torch.randn((imgn, 1)),
        "K": torch.randn((imgn, 1)),
        "alpha": torch.randn((imgn, 1)),
        "beta": torch.randn((imgn, 1)),
        "pitch": torch.randn((imgn, 1)),
        "yaw": torch.randn((imgn, 1)),
        "kq_result": torch.randn((imgn, 2)),
        "light_left": torch.randn((imgn, 3)),
        "light_right": torch.randn((imgn, 3)),
        "cam_o": torch.randn((imgn, 3)),
        "glint_ccs_list": torch.randn((imgn, 2, 3)),
        "pupil_boundary_ccs": torch.randn((imgn, pbn, 3)),
        "pixel_width": torch.randn((imgn, 1)),
        "pixel_height": torch.randn((imgn, 1)),
        "scr_left_top_3d": torch.randn((imgn, 3)),
        "scr_left_bottom_3d": torch.randn((imgn, 3)),
        "scr_right_bottom_3d": torch.randn((imgn, 3)),
    }
    return data, imgn, pbn


def test_pccr():
    data, imgn, pbn = gen_fake_data()
    pccr = PCCRProcess()
    output = pccr(data)

    assert output["angle"].shape == (imgn, 2)
    assert output["center_1"].shape == (imgn, 3)
    assert output["center_2"].shape == (imgn, 3)
    assert output["center"].shape == (imgn, 3)
    assert output["pb"].shape == (imgn, pbn, 3)
    assert output["p"].shape == (imgn, 3)
    assert output["valid_mask"].shape == (imgn, pbn)
    assert output["screen_coords"].shape == (imgn, 2)
