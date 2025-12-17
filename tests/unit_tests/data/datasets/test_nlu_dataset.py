import os

import pytest

from hat.data.datasets.nlu_dataset import (
    NluBasicTokenizer,
    NluLabelProcessor,
    NluMultiTaskDataset,
)
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH

demo_label_path = os.path.join(
    HAT_BUCKET_PATH, "unit_test_data/nlu/demo/label"
)
demo_dict_path = os.path.join(
    HAT_BUCKET_PATH, "unit_test_data/nlu/demo/label/dict.txt"
)
demo_dataset_path = os.path.join(
    HAT_BUCKET_PATH, "unit_test_data/nlu/demo/dataset/trainset"
)


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_nlu_label_processor():
    label_processor = NluLabelProcessor(demo_label_path)
    assert label_processor.domain_label_size == 3


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_nlu_tokenizer():
    tokenizer = NluBasicTokenizer(demo_dict_path)
    assert len(tokenizer.tokenize(["你", "好"])) == 2


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_nlu_dataset():
    label_processor = NluLabelProcessor(demo_label_path)
    tokenizer = NluBasicTokenizer(demo_dict_path)
    nlu_dataset = NluMultiTaskDataset(
        demo_dataset_path, tokenizer, label_processor, 30
    )
    assert nlu_dataset.__len__() == 16
