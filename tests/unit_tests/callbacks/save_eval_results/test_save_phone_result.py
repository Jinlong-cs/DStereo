import torch

from hat.callbacks.save_eval_results.save_phone_result import SavePhoneResult


def test_save_phone_result():
    dummy_batch = {
        "img": torch.randn([2, 3, 128, 128]),
        "labels": torch.randint(0, 3, [2]),
        "aug_type": "test",
        "img_loc": [0, 1],
        "mode": "train",
    }
    kwargs = {"epoch_id": 0, "train_metrics": None}
    dummy_result = [torch.randn([2, 3, 1, 1]), torch.randn(2)]
    save_phone_result = SavePhoneResult(output_dir="./")
    save_phone_result.on_batch_end(dummy_batch, dummy_result, **kwargs)
