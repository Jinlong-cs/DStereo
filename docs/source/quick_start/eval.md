# 挂载数据和验证

HAT 提供了 `train.py`  + `config` 的形式实现模型训练、验证等。下面以 resnet18 分类任务为例进行介绍。


## 准备数据 （Prepare Dataset）
这里使用 HDLTAlgorithm Bucket 上的 ImageNet 数据集。


**方法一(推荐):**

使用make命令一键挂载:
```bash
make basic-data
```

**方法二:**

HAT 提供了相应的 bucket 挂载脚本 `tools/prepare_bucket.py`，
该脚本可以自动挂载 bucket 到开发机，并将 bucket 路径软链接到 `HAT` 目录下：

```bash
python3 tools/prepare_bucket.py --bucket "HDLTAlgorithm"  --mount --create-link

# --bucket          bucket 名字，如 "HDLTAlgorithm"
# --mount           执行挂载
# --create-link     创建软连接
```

执行上述命令后，会把 `HDLTAlgorithm` 挂载到 `/horizon-bucket/HDLTAlgorithm` 目录，并在 `HAT` 目录下创建数据集的软链接 `tmp_data` 和预训练模型的软链接 `tmp_pretrained_models`。

> 如果要使用 HDLTAlgorithm bucket，请确认自己是否已加入以下任一组中：`group-platform-developer`，`group-auto-developer`，`group-aiot-developer` 或 `bucket_HDLTAlgorithm_developer`。

**方法三:**

当然也可参照下面手动把 HDLTAlgorithm Bucket 挂载到开发机：

```bash
# 1. 查看是否有权限访问 HDLTAlgorithm Bucket 
hitc mount show

# 2. 挂载 HDLTAlgorithm Bucket 到开发机 /horizon-bucket/HDLTAlgorithm 目录
hitc mount exec HDLTAlgorithm 
```

HDLTAlgorithm Bucket 成功挂载到开发机后，再软链接数据集和预训练模型例如：

```bash
# 1. 进入 HAT 项目根目录
cd HAT  

# 2. 链接HDLTAlgorithm/data/到 HAT 根目录
ln -s /horizon-bucket/HDLTAlgorithm/data/pack_data ./tmp_data
ln -s /horizon-bucket/HDLTAlgorithm/models/bayes_release_models ./tmp_pretrained_models
```

## 验证

数据挂载好后，可以通过以下方式验证训练和测试功能：

**Training:** 可通过以下命令，训练模型:
```bash
python3 tools/train.py --config examples/classification/resnet18.py --stage "float" -ids "0,1"
```
**Eval:** 模型训练完成后，可通过以下命令，来验证模型指标：
```bash
python3 tools/predict.py --config examples/classification/resnet18.py --stage "float" --ckpt "./tmp_models/resnet18_cls/float-checkpoint-best.pth.tar" -ids "0,1"
```

