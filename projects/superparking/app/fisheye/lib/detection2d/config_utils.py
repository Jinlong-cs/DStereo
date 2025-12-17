from typing import List, Union


def adjust_multitask_sample_weight(
    target_weights: List[Union[float, int]],
    dataset_weights: List[List[Union[float, int]]],
) -> List[float]:
    assert len(target_weights) == len(dataset_weights)
    ret_weights = []
    total_weight = sum(target_weights)
    for tweight, dweight_list in zip(target_weights, dataset_weights):
        d_total = sum(dweight_list)
        tweight = tweight / total_weight
        dweight_list = [w / d_total * tweight for w in dweight_list]
        ret_weights.extend(dweight_list)

    return ret_weights
