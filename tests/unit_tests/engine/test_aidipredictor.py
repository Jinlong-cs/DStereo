import pytest
import torch

from hat.engine import AIDIPredictor

try:
    import aidisdk
    import hatbc
    import perception_proto
    from aidisdk.model import DeviceMeta
    from hatbc.message.frame import CameraFrame, Image
except ImportError:
    DeviceMeta = None
    aidisdk = None
    hatbc = None
    perception_proto = None
    CameraFrame = None
    Image = None


@pytest.mark.skipif(hatbc is None, reason="Need hatbc")
@pytest.mark.skipif(perception_proto is None, reason="Need perception_proto")
@pytest.mark.skipif(aidisdk is None, reason="Need aidisdk")
def test_AIDIPredictor_local():
    img = torch.randn(3, 224, 224)
    frame = CameraFrame(image=Image(data=img))

    infer_engine = AIDIPredictor(
        infer_mode=AIDIPredictor.InferenceMode.InferModelLocal,
        aidi_config="plugins/aidi_inference/configs/aidi_config_test.py",  # noqa
        run_device=DeviceMeta("gpu", 1),
    )

    infer_engine([frame])
