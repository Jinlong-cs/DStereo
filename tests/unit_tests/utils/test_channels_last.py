import torch

from hat.utils.channels_last import convert_memory_format


def test_convert_memory_format():
    batch = {"data1": torch.rand(1, 3, 10, 10), "data2": torch.rand(1, 2)}
    new_batch = convert_memory_format(batch, None)
    assert new_batch["data1"].stride()[-1] == 3
    assert new_batch["data2"].stride()[-1] == 1

    batch = {
        "data1": torch.rand(1, 3, 10, 10),
        "data2": torch.rand(1, 2, 5, 5),
    }
    new_batch = convert_memory_format(batch, ("data1",))
    assert new_batch["data1"].stride()[-1] == 3
    assert new_batch["data2"].stride()[-1] == 1

    batch = {"data1": {"sub_data1": torch.rand(1, 3, 10, 10)}}
    new_batch = convert_memory_format(batch, ("data1",))
    assert new_batch["data1"]["sub_data1"].stride()[-1] == 3

    batch = [{"data1": torch.rand(1, 3, 10, 10)}]
    new_batch = convert_memory_format(batch, None)
    assert new_batch[0]["data1"].stride()[-1] == 3

    batch = [[{"data1": torch.rand(1, 3, 10, 10)}]]
    new_batch = convert_memory_format(batch, None)
    assert new_batch[0][0]["data1"].stride()[-1] == 3
