# 训练 && 预测

参考[config](config.md) 和 [exmaples](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/tree/master/examples) 下的文件，配置config文件。
当配置好config文件后，我们就可以使用以下命令来进行不同阶段的训练和预测：

## 训练

`HAT` 的 `tools` 下提供训练工具，可使用以下命令启动不同阶段的训练：

```bash
python3 tools/train.py --config /PATH/TO/CONFIG --stage "float" 
python3 tools/train.py --config /PATH/TO/CONFIG --stage "calibration"
python3 tools/train.py --config /PATH/TO/CONFIG --stage "qat"
python3 tools/train.py --config /PATH/TO/CONFIG --stage "int_infer"
```


## 预测

`HAT` 的 `tools` 下提供预测工具，可使用以下命令启动不同阶段的预测：

```bash
python3 tools/predict.py --config /PATH/TO/CONFIG --stage "float"
python3 tools/predict.py --config /PATH/TO/CONFIG --stage "calibration"
python3 tools/predict.py --config /PATH/TO/CONFIG --stage "qat"
python3 tools/predict.py --config /PATH/TO/CONFIG --stage "int_infer"
```

## 仅进行浮点训练

如果用户仅需要进行浮点训练及预测，只需要依次执行以下命令即可：

```bash
python3 tools/train.py --config /PATH/TO/CONFIG --stage "float" 
python3 tools/predict.py --config /PATH/TO/CONFIG --stage "float"
```

相关的 `config` 文件中只需要添加和浮点训练相关的配置，而无需包含任何量化训练及定点推理的设置。具体可以参考[浮点配置示例](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/tree/master/examples/float_training/mobilenetv1.py)。在模型实现代码中，无需添加和 `qconfig` 设置相关的接口。

## 通过命令行覆盖config参数

HAT 支持通过命令行修改config 参数，使用方式，在 --opts 后面加上 (key, value) 键值对即可：

```bash
python3 tools/train.py --stage float --config examples/classification/resnet18.py --opts "key1" "value1" "key2" "value2"
```

其中，key 是要修改的参数名，value 为想要传入的值。

例如配置文件 config.py:

```python
task_name = "resnet_cls"
num_classes=1000
devices = [0, 1, 2, 3]
data_shape = (3, 224, 224)
model = dict(
    backbone="resnet18",
    num_classes=num_classes,
)
```
> 注: 本文档中的示例仅为说明使用 --opts 时的不同类型参数的正确书写格式， 示例 config 中的参数、字段配置等，可能与项目中实际 config 略有不同。

* key 可以支持多级参数的修改，比如下面命令:
    ```bash
    python3 tools/train.py --stage float --config config.py --opts model.backbone "'resnet50'"
    ```

    上面这条命令，可以把 config 中 `model.backbone` 字段的值修改成`"resnet50"`

* Value 可以是数值、字符串(str)、列表(list)、元组(tuple)形式，但不支持 dict 形式。

    比如下面命令可以把 config 中 model.num_classes 字段的值修改成10

    ```bash
    python3 tools/train.py --stage float --config config.py --opts num_classes 10
    ```

    但是由于解析机制的原因，在要传入 value 是 tuple 或 list 形式时，需要对 tuple 或 list 加上引号，例如:
    ```bash
    python3 tools/train.py --stage float --config config.py --opts data_shape (3, 112, 112)  # 这样会解析失败
    python3 tools/train.py --stage float --config config.py --opts data_shape "(3, 112, 112)" # 这样是正确的
    ```

    如果要传入的 value 是 str 形式的，需要额外加上引号,例如:
    ```bash
    python3 tools/train.py --stage float --config config.py --opts model.backbone resnet18 # 这样会解析不正确
    python3 tools/train.py --stage float --config config.py --opts model.backbone "resnet18" # 这样会解析不正确
    python3 tools/train.py --stage float --config config.py --opts model.backbone "'resnet18'" # 这样是正确的
    ```

## 通过命令行设置环境变量

HAT 支持通过命令行设置环境变量，可在config.py中使用等，使用方式如下：
    
```bash
python3 tools/train.py --stage float --config config.py --sample-env1=xxx1 --sample-env2 xxx2  --sample_env3=xxx3 
```

当在`train.py` 中传入无效参数时，会将其识别为自定义环境变量参数，并将key中的 `-` 转换成 `_`, 小写字母全部改成大写字母，
如上命令行，最终设置的环境变量参数如下：
```bash
SAMPLE_ENV1 xxx1
SAMPLE_ENV2 xxx2
SAMPLE_ENV3 xxx3
```
