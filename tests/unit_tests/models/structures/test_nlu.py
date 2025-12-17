import os

import pytest
import torch

from hat.data.datasets.nlu_dataset import NluBasicTokenizer, NluLabelProcessor
from hat.models.backbones.nlu_tcn import NluMultiTaskBackbone
from hat.models.structures.nlu import NluModel
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
def test_nlu_model():
    label_processor = NluLabelProcessor(demo_label_path)
    tokenizer = NluBasicTokenizer(demo_dict_path)
    nlu_backbone = NluMultiTaskBackbone(tokenizer, label_processor)
    loss = torch.nn.CrossEntropyLoss(reduction="none")
    nlu_model = NluModel(nlu_backbone, loss)

    batch_size, seq_length = 10, 15
    fake_ids = torch.randint(tokenizer.vocab_size, (batch_size, seq_length))
    fake_domain_labels = torch.randint(
        label_processor.domain_label_size, (batch_size,)
    )
    fake_intent_labels = torch.randint(
        label_processor.intent_label_size, (batch_size,)
    )
    fake_slots_labels = torch.randint(
        label_processor.slots_label_size, (batch_size, seq_length - 2)
    )
    fake_pad_masks = torch.ones((batch_size, seq_length - 2)).bool()
    fake_weights = torch.ones(batch_size)
    fake_data = {
        "batch_query": None,
        "batch_ids": fake_ids,
        "batch_pad_masks": fake_pad_masks,
        "batch_domain_labs": fake_domain_labels,
        "batch_intent_labs": fake_intent_labels,
        "batch_ner_labs": fake_slots_labels,
        "batch_sample_weights": fake_weights,
    }
    preds, targets, total_loss = nlu_model(fake_data)
    assert len(preds[0]) == batch_size
