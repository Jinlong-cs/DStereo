import os

from hat.utils.logger import rank_zero_warn

try:
    if os.getenv("NO_HDFLOW", "0") == "1":
        os.environ.pop("NO_HDFLOW")
        raise ImportError("hdflow not allowed")

    from hdflow.dataset_verify.partition_manager import PartitionManager

    partition_disable = False
except ImportError as e:
    rank_zero_warn(e)
    partition_disable = True
    PartitionManager = None


def parse_by_partition(datapaths, partitions, training_step, log_dir):
    if partition_disable:
        rank_zero_warn("partition not supported in local hdflow version")
        return datapaths
    manager = PartitionManager(partitions, training_step)
    datapaths = manager.parse(datapaths)
    manager.dump_stats(log_dir)
    return datapaths
