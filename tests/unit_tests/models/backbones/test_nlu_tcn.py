import os

import pytest
import torch

from hat.data.datasets.nlu_dataset import NluBasicTokenizer, NluLabelProcessor
from hat.models.backbones.nlu_tcn import NluMultiTaskBackbone
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH

demo_label_path = os.path.join(
    HAT_BUCKET_PATH, "unit_test_data/nlu/demo/label"
)
demo_dict_path = os.path.join(
    HAT_BUCKET_PATH, "unit_test_data/nlu/demo/label/dict.txt"
)

try:
    import torchcrf
except ImportError:
    torchcrf = None


@pytest.mark.skipif(torchcrf is None, reason="need torchcrf")
@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_nlu_backbone():
    label_processor = NluLabelProcessor(demo_label_path)
    tokenizer = NluBasicTokenizer(demo_dict_path)
    nlu_backbone = NluMultiTaskBackbone(tokenizer, label_processor)
    input_embed = torch.randn(5, 512, 1, 30)
    domain_out, intent_out, slots_out = nlu_backbone(input_embed)
    assert domain_out.shape[0] == 5
