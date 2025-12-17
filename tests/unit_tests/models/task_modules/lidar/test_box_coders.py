import torch

from hat.models.task_modules.lidar import GroundBox3dCoder


def test_groundbox3dcoder():

    boxes = torch.randn([10, 7])
    anchors = torch.randn([10, 7])

    box_coder = GroundBox3dCoder(n_dim=7)

    encoded_box = box_coder.encode(boxes, anchors)
    assert encoded_box.shape == (10, 7)

    decoded_box = box_coder.decode(boxes, anchors)
    assert decoded_box.shape == (10, 7)
