(hat-compile-model)=
# 如何编译模型

HAT 集成了编译定点模型的能力，将 pytorch 的模型编译为可以上板运行的形式。用户可以选择两种方式编译模型。

## 1. 输入为 pt 模型或 hbir 文件


此方式使用 `tools/deploy/compile_standalone.py` 脚本，依赖定点模型 trace 出的 pt 文件或 hbir 文件，但不依赖 HAT，可脱离 HAT 单独使用。

>注意：使用的 pt 模型中只能包含可以上板的定点算子。

命令如下：

```bash
python3 tools/deploy/compile_standalone.py {$PT_FILE} --input-size {$INPUT_SIZE} --name {$NAME} --opt O2 --march {$MARCH}
```

其中，各参数含义如下：
- `in_file` (str):（必须）pt 或 hbir 文件路径
- `--input-size` (str):（必须）输入数据的 shape 大小，`[Nx]BxCxHxW` 格式，即 N 个 `BxCxHxW` 的列表，多输入之间以 ^ 分割。比如："1x3x224x224^1x3x224x224"
- `--input-key` (str):（可选）输入数据的名称, 默认 "img"。多输入之间以 ^ 分割
- `--input-source` (str):（可选）编译输入源，默认 "pyramid"
- `--input-layout` (str): （可选）模型输入维度格式信息，hbdk 支持的格式为 ["NHWC", "NCHW", "BPU_RAW"]
- `--output-layout` (str): （可选）模型输出维度格式信息
- `--output` (str):（可选）编译产物(hbir、hbm)输出路径。默认 "tmp_compile"
- `--name` (str):（可选）设置生成的 hbm 文件名称
- `--opt` (str):（可选）编译优化级别，可设置 "O0"、"O1"、"O2"、"O3"，"ddr", "fast"，默认 "O0"。（注：数值越大，上板优化效果越好，但编译过程也耗时越长，一般将 `opt` 参数设置为 `O2`，若对模型上板性能要求较高，也可以设置为 `O3`，但是编译速度会比较慢）
- -`-march` (str):（必须）设置芯片架构，参数可以选择 `bayes` 或 `bernoulli2`，默认 `bayes`
- `--hbir-only`:（可选）只导出 hbir 文件
- `--perf-only`: （可选）只进行 perf
- `--debug`：（可选）已 debug 模式进行编译，会在 perf 结果中显示 layer 的详细信息


执行命令，编译完成后，编译出的模型存放在设置的 output (默认tmp_compile） 路径下，包含以下几个文件：
- model_opt_{$OPT}.hbm : 编译出的模型文件（hbm）
- model_opt_{$OPT}.hbir: 编译过程中使用的模型的中间表示（hbir）
- {\$NAME}.html, {\$NAME}.json：以不同格式保存的模型上板相关的信息如运行速度、资源占用等


## 2. 输入为 config 文件

此方式使用 `tools/deploy/compile_perf.py` 脚本，依赖：已有模型的 config 文件，即依赖 HAT。需要编译的模型来自于 `int_infer_trainer` 中定义的 `deploy_model` ，命令如下：

```bash
python3 tools/deploy/compile_perf.py -c {$CONFIG_FILE}
```

可传入参数含义如下：
- `-c` (str): （必须） config 文件路径。
- `--out-dir` (str): （可选），编译产物的保存路径，含义同 config 文件 compile_cfg 中的 "out_dir" 参数。若传入此参数，则会覆盖 compile_cfg 中的设置
- `--opt` (str): (可选) 编译优化级别，含义同 config 文件 compile_cfg 中的 "opt" 参数。若传入此参数，则会覆盖 compile_cfg 中的设置
- `--jobs` (int):（可选）默认4，编译使用的线程数。（设置为 0 时，会使用所有可用的硬件资源）
- `--save_qresults` (int)：（可选）含义同 config 文件 compile_cfg 中的 "save_qresults" 参数。若传入此参数，则会覆盖 compile_cfg 中的设置
- `--ckpt` (str): （可选）定点模型 checkpoint 路径。不传入此参数时，会默认用 config 文件中设置的 qat checkpoint 转为定点的方式，来完成定点模型参数的初始化

上述命令会从 config 文件的 `compile_cfg` 字段获取编译相关配置，举例说明如下：

```python
import horizon_plugin_pytorch

compile_cfg = dict(
    march=horizon_plugin_pytorch.march.March.BAYES, # 可以选择 BAYES 或 BERNOULLI2
    name="example_model",  # perf 信息的文件名
    out_dir="perf_results",  # perf 信息的输出路径
    hbm="tmp_compile/model.hbm",  # 编译出的模型文件路径
    layer_details=True,  # 是否输出每层的性能情况
    input_source=["pyramid"],  # 模型上板时的输入来源，一般选 pyramid 即可
    opt="O2",  # 优化等级，可以选择 O0, O1, O2, O3，优化程度从低到高
    save_qresults=False, # 是否将定点模型中每一层的输出以numpy的形式存储下来，方便在定点模型和编译器模拟器结果对不齐时，辅助编译器进行结果比对，默认为False
)
```

编译完成后，会生成以下几个文件：
- tmp_compile/model.hbm：编译出的模型文件
- tmp_compile/model.hbir：编译过程中使用的模型的中间表示
- perf_results/example_model.json, perf_results/example_model.html：以不同格式保存的模型上板相关的信息如运行速度、资源占用等
- tmp_compile/plugin_quantized_result.npy：将定点模型中每一层的结果存储在npy文件中，辅助编译器进行结果比对
