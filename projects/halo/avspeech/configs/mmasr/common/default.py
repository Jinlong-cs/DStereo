# flake8: noqa

from aidisdk.utils import running_in_cluster

tendis_kwargs = dict(
    host="aidi-kv-cluster-01.hogpu.cc",
    port=7616,  # 原来是 7617
    db=0,
    socket_connect_timeout=5000,
    password="pB9cA2eN1eD1lJ0",
    socket_keepalive=False,
)

bucket = "/bucket/input" if running_in_cluster() else "/horizon-bucket"


global_keys = [
    "current_config",
    "training_step",
    "model_prefix",
    "task_name",
    "batch_size",
    "num_workers",
    "step_log_freq",
    "epoch_log_freq",
    "fbank_size",
    "bpe_model_path",
    "symbol_table",
    "symbol_table_path",
    "vocab_size",
]
