(hat-config)=
# config 文件介绍

使用 `HAT` 算法包训练模型通常只需使用一条命令就可以了，即：`python3 tools/train.py --stage TRAINING_STEPS --config /PATH/TO/CONFIG`，验证模型通常也只需要使用一条命令，即：`python3 tools/predict.py --stage TRAINING_STEPS --config /PATH/TO/CONFIG`，其中 `/PATH/TO/CONFIG` 
就是模型训练对应的 `config` 文件，它负责定义了模型结构、数据集加载、以及整套的训练和验证的流程。  

这篇教程通过介绍 `config` 文件中的一些关键字，让用户对 `config` 中的内容以及作用有个大致的了解。  

我们把这些关键字分成 **核心关键字** 和 **推荐关键字** 两种类型。所谓 **核心关键字** 是指会被 `tools` 下面的脚本直接调用的关键字，因此是必须写在 `config` 文件中的；而 **推荐关键字** 通常是核心关键字下面的一些 `key` 值，你可以在 `config` 下面直接定义，从而方便 **核心关键字** 的定义，也可以不用在 `config` 中定义。

## 核心关键字
`device_ids`: 模型训练使用的 `gpu` 列表。  

`VERSION`: config的版本号，用于解决config训练或预测走不同的逻辑。

`cudnn_benchmark`: 是否打开cudnn benchmark。通常默认为 `True` 。   

`seed`: 是否设置随机数种子。通常默认为 `None` 。

`gpu_affinity`: 是否使用gpu亲和力的配置。通常默认为 `none`。可选范围为`none`, `socket`, `socket_single`, `socket_single_unique`, `socket_single_interleaved`, `socket_unique_contiguous`。其中推荐的设置是`socket_unique_contiguous`。

`log_rank_zero_only`: 简化多卡训练时的日志打印，只在第0卡上输出日志。通常默认为 `True` 。  

`march`：表示模型最终部署到什么架构的芯片上，默认为 `March.BAYES` 。 

`deploy_model`: 参与 `deploy` 过程的模型结构，主要用于模型编译。和 `model` 相比，大多数情况下只需要把损失函数以及后处理部分设置为 `None` 即可。

`deploy_inputs`: 模型编译过程的模拟输入。不用关心具体的数值，只要保证格式满足输入要求即可。

`{stage}_trainer`：表示的是 `{stage}` 阶段的训练流程配置。例如当你需要做 `float` 阶段训练的时候，需要在 `config` 中配置 `float_trainer` 关键字；当你需要做 `qat` 阶段训练的时候，需要在 `config` 中配置 `qat_trainer` 关键字。这些和训练流程相关的关键字，内部配置几乎一致，例如 `type` 值为 `distributed_data_parallel_trainer`，表示分布式训练，`model` 表示参与训练的模型结构，`model_convert_pipeline` 表示模型的 `convert` 过程，`data_loader` 表示参与训练的数据集，`optimizer` 表示优化器，`callbacks` 表示训练过程中进行的一些列操作，例如模型保存、学习率更新等，`train_metric` 表示训练过程的指标验证，`val_metric` 表示每个 `epoch` 结束模型验证过程的指标验证。更加具体的参数含义可以参考我们的 `API` [文档](http://model.aidi.hobot.cc/api/docs/3240/HAT/1.1.0.dev202206130413-77c7d5f/html/api_reference/engine.html#hat.engine.DistributedDataParallelTrainer)。如何配置参数可以参考 [mobilenetv1.py](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/blob/master/examples/classification/mobilenetv1.py) 中的写法。  

`compile_cfg`: 编译相关的配置。`out_dir` 表示编译生成的 `hbm` 文件（部署模型）的输出路径。    

`{stage}_predictor`：该关键字当前只有使用 `tools/predict.py` 脚本的时候才会调用到。表示的是 `{stage}` 阶段的验证流程配置。例如当你需要做 `float` 阶段验证的时候，需要在 `config` 中配置 `float_predictor` 关键字；当你需要做 `qat` 阶段验证的时候，需要在 `config` 中配置 `qat_predictor` 关键字；当你需要做 `int_infer` 阶段验证的时候，需要在 `config` 中配置 `int_infer_predictor` 关键字。这些和验证流程相关的关键字，内部配置几乎一致。例如 `type` 值为 `Predictor`，`model` 表示参与验证的模型结构，`model_convert_pipeline` 表示模型的 `convert` 过程，`data_loader` 表示验证的数据集，`callbacks` 表示验证过程中进行的一些列操作，`val_metric` 表示模型验证过程的指标打印。更加具体的参数含义可以参考我们的 `API` [文档](http://model.aidi.hobot.cc/api/docs/3240/HAT/1.1.0.dev202206130413-77c7d5f/html/api_reference/engine.html#hat.engine.Predictor)。如何配置参数可以参考 [mobilenetv1.py](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/blob/master/examples/classification/mobilenetv1.py) 中的写法。 

`onnx_cfg`: 导出onnx模型的相关配置。通常 `model` 为 `deploy_model` ， `input` 为 `deploy_input` , `stage` 和 `model_convert_pipeline` 根据要导出的阶段进行选择。

## 推荐关键字
`model`: 参与 `training` 过程中的模型结构。`type` 表示模型的类型，如 `Classifier`、`Segmentor`、`RetinaNet`等等，分别对应分类、分割、检测中的某一类模型。它会在使用过程中被 `build` 成具体的类，余下的参数都是用于初始化这个类。 

`data_loader`: 训练阶段的数据集加载流程。它的 `type` 是一个具体的类 `torch.utils.data.DataLoader` ，余下的参数都是用于初始化这个类。相关参数的含义也可以参考 `pytorch` 官网提供的接口文档。这里 `dataset` 表示读取某个具体的数据集，例如`ImageNet`、`MSCOCO`、`VOC`等等，它的 `transforms` 表示在数据读取过程中添加的数据增强操作。

`val_data_loader`: 验证模型性能阶段的数据集加载流程。和 `data_loader` 不同的地方在于 `data_path` 不同，以及去掉了 `transforms` 的过程和 `sample` 的过程。  

`batch_processor`: 模型在训练过程中每个迭代 `step` 进行的操作，包括前向计算、梯度回传、参数更新等等。如果包含 `batch_transforms` 参数，表示一些数据增强的操作是在 `gpu` 上进行的，这可以大大加快训练速度。  

`val_batch_processor`: 模型在验证过程中每个迭代 `step` 进行的操作，只包含前向计算。  

`metric_updater`: 模型训练过程中更新指标的方法，这个指标是用来验证训练的模型性能是否在提升。它通常是和 `{stage}_trainer` 下面的 `train_metrics` 配合着使用。`train_metrics` 是具体的指标形式，`metric_updater` 只是提供一种更新方法。  

`val_metric_updater`: 训练出来的模型在验证性能的过程中更新指标的方法，这个指标用来验证最终训练出来的模型性能到底如何。它通常是和 `{stage}_trainer` 或者 `{stage}_predictor` 下面的 `val_metrics` 配合着使用，和 `metric_updater` 同理。  

`ckpt_callback`：模型保存的 `callback`。通常在模型训练过程中或结束的时候都需要保存模型。  

`val_callback`：验证模型性能的 `callback`。通常在模型训练过程中或者结束的时候，需要验证模型的性能。  

## 如何配置
这里主要介绍数据类型为 `dict` 的 **核心关键字** 的配置。  

数据类型为 `dict` 的 **核心关键字** 可以分为两种，包含 `type` 的，例如 `{stage}_trainer` 、 `{stage}_predictor`等，和不包含 `type` 的，例如 `compile_cfg`。  

它们的区别在于包含 `type` 的 **核心关键字** 本质可以看作是一个 `class`，它的 `type` 值可以是一个 `string` 变量，也可以是一个具体的 `class`。如果是 `string` 变量，在程序运行中同样会被 `build` 成一个相应的 `class`。这个 `dict` 中除掉 `type` 之外的其他 `keys` 的值都用于初始化这个 `class` 。和 **核心关键字** 类似，这些 `keys` 的值可以是一个数值，也可以是一个包含 `type` 变量的 `dict`, 也就是 **推荐关键字** 。例如 `{stage}_trainer` 中的 `data_loader`，以及这个 `data_loader` 下面的 `dataset`。更详细的写法可以参考注册机制相关的 [文档](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/blob/master/docs/source/tutorials/module_development/registry.md)。  

对于没有 `type` 变量的 **核心关键字** 来说，它就是一个普通类型的 `dict` 变量，代码在运行过程中会通过其 `keys` 获取对应的 `values`。
