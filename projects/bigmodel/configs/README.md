
## Requirements

Local install mmcv for local training.

```bash
# CUDA_version support cu111, cu102
# TORCH_version support torch1.10.x, torch.1.9.x
# For example: 
    # CUDA_Version = {cu111: cu111, cu102: cu102}
    # TORCH_version = {torch1.10.x: torch1100, torch.1.9.x: torch191}
    
pip3 install --upgrade --no-cache-dir \
mmcv_full==1.4.2 -f \
https://art-internal.hobot.cc/artifactory/custom-algo-pypi/mmcv/cu111/torch1100 \
--trusted-host art-internal.hobot.cc \
-i https://pypi.hobot.cc/simple \
--extra-index-url=https://pypi.hobot.cc/hobot-local/simple \
```


## QuickStart

Local training

```bash
# please verify mmcv is installed in your environment
export HAT_USE_CHECKPOINT="1"
python tools/train.py \
--config projects/bigmodel/configs/sparse4d/entry.py \
--stage "float" \
-ids "0,1"
```


AIDI training

```
sh projects/bigmodel/tools/submit.sh
```

注意以下参数与集群训练配置相关：

`common.py` 中的

- `local_or_remote_debug`: `True`表明在开发机本地或者集群中debug，在集群中正常训练中应设置为`False`。
- `num_machines`: 机器数。
- `num_gpus_per_machine`: 每台机器使用的GPU数，本地默认为`2`，集群默认为`8`。
- `project_id`: 项目号。
- `enable_temporal_fusion`: `False`为单帧模型，`Ture`为时序模型。
- `do_experiment`: `True`为快速验证实验，`False`为正式发版设置。

`projects/bigmodel/tools/submit.sh` 中的

- `cluster`: 训练集群队列名。

## Evaluation

Local evaluation

``` bash
python tools/predict.py \
--config projects/bigmodel/configs/sparse4d/eval_entry.py \
--stage "float" \
-ids "0"
```

`eval_entry.py` 中关键字段：

- `eval_dataset_id`: aidi评测集dataset_id。
- `eval_ckpt_path`: 待评测的模型checkpoint。
