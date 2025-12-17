import torch

from hat.models.task_modules.PCCR_gaze.screencoord import ScreenCoordinate


def gen_fake_data():
    imgn = 8
    data = {
        "gaze": torch.randn((imgn, 3)),
        "origin": torch.randn((imgn, 3)),
        "lb": torch.randn((imgn, 3)),
        "lt": torch.randn((imgn, 3)),
        "rb": torch.randn((imgn, 3)),
        "pixel_width": torch.randn((imgn, 1)),
        "pixel_height": torch.randn((imgn, 1)),
    }
    return data, imgn


def test_screen():
    data, imgn = gen_fake_data()
    screen = ScreenCoordinate()
    output = screen(**data)
    assert output.shape == (imgn, 2)
