import numpy as np
import torch

from hat.models.task_modules.sparse4d import InstanceBank


def test_instance_bank():
    num_anchor = 900
    embed_dims = 256
    state_dims = 10
    num_temp_instances = 600
    batch_size_1 = 4
    batch_size_2 = 2

    anchor = np.random.uniform(size=(num_anchor, state_dims)).tolist()
    # import pdb; pdb.set_trace()
    instance_bank = InstanceBank(
        num_anchor=num_anchor,
        embed_dims=embed_dims,
        anchor=anchor,
        num_temp_instances=num_temp_instances,
        max_queue_length=10,
    )
    metas = {"timestamp": torch.tensor([1] * batch_size_1)}
    (
        instance_feature,
        anchor,
        cached_feature,
        cached_anchor,
        time_interval,
    ) = instance_bank.get(batch_size_1, metas)
    assert instance_feature.shape == (batch_size_1, num_anchor, embed_dims)
    assert anchor.shape == (batch_size_1, num_anchor, state_dims)
    assert cached_feature is None
    assert cached_anchor is None
    assert time_interval.shape == (batch_size_1,)

    confidence = torch.randn(instance_feature.shape)
    instance_bank.cache(instance_feature, anchor, confidence, metas)
    (
        instance_feature,
        anchor,
        cached_feature,
        cached_anchor,
        time_interval,
    ) = instance_bank.get(batch_size_1, metas)
    instance_feature, anchor = instance_bank.update(
        instance_feature, anchor, confidence
    )
    assert instance_feature.shape == (batch_size_1, num_anchor, embed_dims)
    assert anchor.shape == (batch_size_1, num_anchor, state_dims)
    (
        instance_feature,
        anchor,
        cached_feature,
        cached_anchor,
        time_interval,
    ) = instance_bank.get(batch_size_1, metas)
    assert cached_feature.shape == (
        batch_size_1,
        num_temp_instances,
        embed_dims,
    )
    assert cached_anchor.shape == (
        batch_size_1,
        num_temp_instances,
        state_dims,
    )

    metas_2 = {"timestamp": torch.tensor([1] * batch_size_2)}
    (
        instance_feature,
        anchor,
        cached_feature,
        cached_anchor,
        time_interval,
    ) = instance_bank.get(batch_size_2, metas_2)
    assert instance_feature.shape == (batch_size_2, num_anchor, embed_dims)
    assert anchor.shape == (batch_size_2, num_anchor, state_dims)
    assert cached_feature is None
    assert cached_anchor is None
    assert time_interval.shape == (batch_size_2,)
