from dataclasses import dataclass
from typing import List, Optional, Union

import yaml
from dataclasses_json import DataClassJsonMixin
from hatbc.aidi.dmp_client import DmpClient
from hatbc.auto_dp.database import DEFAULT_CONFIG_PATH
from hatbc.utils import Enum
from hdflow.data.anno_data_manage.data_stats import (
    AnnoRecKeyIter,
    AutoDpMetaGenerator,
    EvalSetKeyIter,
    LabelJsonMetaGenerator,
)

database_config = yaml.safe_load(open(DEFAULT_CONFIG_PATH, "r"))


class StatisticSource(Enum):
    leaderboard_id = "leaderboard_id"
    densebox_json = "densebox_json"
    densebox_pbrec = "densebox_pbrec"


@dataclass
class StatisticItem(DataClassJsonMixin):
    source_type: StatisticSource
    values: Union[str, int, List[int], List[str]]
    item_name: str
    collect_keys: List[str]
    task: Optional[str] = "detection"  # "detection" or segmentation


def iters_from_source(statistic_items: List[StatisticItem]):
    dmp_client = DmpClient(token=database_config["token"])

    image_key_iters = list()
    meta_generators = list()
    valid_items = list()
    for statistic_item in statistic_items:
        if statistic_item.source_type == StatisticSource.leaderboard_id:
            image_key_iters.append(
                EvalSetKeyIter(
                    eval_set_ids=statistic_item.values,
                    datasets_paths=None,
                    # filter by this
                    task_types=None,
                    batch_size=100,
                    dmp_client=dmp_client,
                )
            )
            if statistic_item.task == "detection":
                meta_generators.append(
                    LabelJsonMetaGenerator(
                        image_meta_keys=statistic_item.collect_keys
                    )
                )
            elif statistic_item.task == "segmentation":
                meta_generators.append(
                    AutoDpMetaGenerator(
                        image_meta_keys=statistic_item.collect_keys,
                        pack_meta_keys=statistic_item.collect_keys,
                    )
                )
            valid_items.append(statistic_item)
        elif statistic_item.source_type in [
            StatisticSource.densebox_json,
            StatisticSource.densebox_pbrec,
        ]:
            # TODO
            image_key_iters.append(
                AnnoRecKeyIter(
                    anno_paths=statistic_item.values,
                    batch_size=100,
                    dmp_client=dmp_client,
                )
            )
            meta_generators.append(
                AutoDpMetaGenerator(
                    image_meta_keys=statistic_item.collect_keys,
                    pack_meta_keys=statistic_item.collect_keys,
                )
            )
            valid_items.append(statistic_item)
    return valid_items, image_key_iters, meta_generators, dmp_client
