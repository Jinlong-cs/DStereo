import torch
import yaml

from hat.models.backbones.hrnet import HRNet


def test_HRNet():
    config = dict(
        yaml.safe_load(open("tmp_orig_data/landmark/hat_test/hrnet.yaml", "r"))
    )
    hrnet = HRNet(config["hrnet_w18"])
    data = torch.randn(1, 3, 256, 256)
    output = hrnet(data)
    assert output.shape == (1, 270, 64, 64)
