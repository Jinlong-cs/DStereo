该目录下为hat的示例文件， 其中：

`classification`: 分类示例模型，包括浮点训练(预测)，量化训练(预测)，转定点模型(预测)。详见[模型训练](http://model.aidi.hobot.cc/api/docs/HAT/latest/html/tutorials/index.html#id2)

`detection`: 检测示例模型，功能同 `classification`。详见[模型训练](http://model.aidi.hobot.cc/api/docs/HAT/latest/html/tutorials/index.html#id2)

`aidi_exp.py`：aidi 实验管理示例。详见[AIDI实验管理](http://model.aidi.hobot.cc/api/docs/HAT/latest/html/tutorials/aidi_platform/aidi_expmodel.html?highlight=aidi_exp)

`export_onnx.py`：导出onnx模型示例，使用方式见[如何导出ONNX](http://model.aidi.hobot.cc/api/docs/HAT/latest/html/tutorials/common_tools/export_onnx.html)。

`export_trt.py`：导出tensorrt示例，使用方式见[如何导出tensorrt](http://model.aidi.hobot.cc/api/docs/HAT/latest/html/tutorials/common_tools/export_tensorrt.html)。

`inference.py`：inference示例，详见[inference](http://model.aidi.hobot.cc/api/docs/HAT/latest/html/tutorials/model_application/implementation.html?highlight=inference)

`model_profile.py`：量化训练profile工具使用示例，可以用来定位量化训练中的精度问题。详见[量化训练Debug工具](http://model.aidi.hobot.cc/api/docs/HAT/latest/html/tutorials/common_tools/model_profiler.html)。

`training_profile.py`: 训练profile工具使用示例，可以用来定位训练过程中的内存，速度等问题。详见[Profile工具使用](https://horizonrobotics.feishu.cn/wiki/wikcnVyH4wU0Bhn8aakr2sdxJSG)

`model_compile.py`: 定点模型编译示例，可以编译出可以上板的 `hbm` 文件。具体使用方式见[如何编译模型](http://model.aidi.hobot.cc/api/docs/HAT/latest/html/tutorials/common_tools/compile.html?highlight=compile_perf)



更多的示例模型见：
[TOOLCHAIN](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/tree/master/projects/toolchain/configs)

Model Zoo: `/horizon-bucket/HDLTAlgorithm/models/bayes_release_models`