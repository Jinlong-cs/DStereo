import torch

from hat.models.task_modules.occflow.decoder.occflow_head import OccflowHead


def test_OccflowHead():
    num_class = 1
    in_channels = 128
    output_channel = 16 * num_class

    model = OccflowHead(
        num_class=num_class,
        in_channels=in_channels,
    )

    input = torch.randn((1, in_channels, 256, 256))

    outputs = model([input])

    assert "occ_preds" in outputs
    assert outputs["occ_preds"].shape == (1, output_channel, 256, 256)

    assert "flow_preds" in outputs
    assert outputs["flow_preds"].shape == (1, output_channel, 256, 256)
