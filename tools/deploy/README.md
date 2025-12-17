该目录下为部署工具， 包括导出onnx模型、tensorrt模型、编译上板模型等工具，其中：

`compile_perf.py` 和 `compile_standalone.py` 均为编译上板模型的工具，具体使用方式见[如何编译模型](http://model.aidi.hobot.cc/api/docs/HAT/latest/html/tutorials/common_tools/compile.html?highlight=compile_perf)。

`export_onnx.py`: 导出onnx模型工具，使用方式见[如何导出ONNX](http://model.aidi.hobot.cc/api/docs/HAT/latest/html/tutorials/common_tools/export_onnx.html)。

`export_tensorrt.py`：导出tensorrt工具，使用方式见[如何导出tensorrt](http://model.aidi.hobot.cc/api/docs/HAT/latest/html/tutorials/common_tools/export_tensorrt.html)。

`model_checker.py`：模型编译检查工具，用于检查模型是否能编译。[如何使用模型检查工具](http://model.aidi.hobot.cc/api/docs/HAT/latest/html/tutorials/model_training/develop_config.html?highlight=model_checker#model-checker)。

`pack_hbm.py`：hbm pack工具。
