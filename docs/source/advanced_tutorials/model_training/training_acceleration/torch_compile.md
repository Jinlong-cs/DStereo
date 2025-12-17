# 如何开启 TORCH COMPILE


`torch.compile` 是 torch2.0 中引入的加速 PyTorch 代码的最新方法，`torch.compile` 通过 JIT 将 PyTorch 代码编译成优化的内核，使 PyTorch 代码运行得更快，详见 [pytorch blog](https://pytorch.org/get-started/pytorch-2.0/)。

HAT 中已经支持了 `torch.compile` 的使用，用户只需要在定义 config 文件中的 `xxx_trainer` 或 `xxx_predictor` 中加入 `Torch2Compile` converter 即可，例如：

```python
# examples/torch_compile.py

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=[],
    # 添加 Torch2Compile，使用 torch.compile()
    compiler=dict(
        type="Torch2Compile",
        # 1. skip module，设置不编译的 module 
        skip_modules=[".*loss.*"],
        regex=True,
                
        # 2. 设置 torch._dynamo.config 的参数
        dynamo_cfg=dict(
            log_level=logging.WARNING,
            optimize_ddp=True,
            # cache_size_limit=128
        ),
        # 3. `torch.compile()` 接口参数
        # 可参照：https://pytorch.org/docs/stable/generated/torch.compile.html#torch.compile 
        fullgraph=False,
        dynamic=False,
        backend="inductor",
        mode="default",
    ),
    data_loader=...,
    optimizer=...,
    batch_processor=...,
    ...
)
```

其中，`Torch2Compile` 可设置的参数主要包括三部分：

1. `skip_modules` 的参数设置，该部分参数是为了指定在 `torch.compile` 时跳过哪些 module（即不做 compile），主要包括：
   * `skip_modules`：指定不 compile 的 module，可以是正则表达式（比如 `[".*loss.*"]`）,也可以是 module name（比如 `[backbone]`）;
   * `regex`: 根据传入的 `skip_modules` 类型设置，如果是正则表达式，则需设为 `True`（默认值`True`），如果传入的是 module name，则需设置为 `False`;

2. `torch._dynamo.config` 的设置参数，详细内容可查看 [pytorch 代码](https://github.com/pytorch/pytorch/blob/release/2.0/torch/_dynamo/config.py), 这里主要介绍三个参数：
   * `log_level`: torch dynamo 的日志级别。测试中发现使用 `torch.compile` 后，会输出大量 log，可能影响正常训练 log 的查看，因此，HAT 中默认设置了 `torch._dynamo.config.log_level=logging.WARNING`，用户可根据需要，设置为 `INFO` 或 `DEBUG`;
   * `optimize_ddp`: TorchDynamo + Eager DDP 的组合会降低 DDP 的训练速度，因此，torch 开发了 `DDPOptimizer`，实现 TorchDynamo 下 DDP 的训练加速，具体内容可查看 [pytorch discuss](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860), 可通过 `optimize_ddp` 设置是否使用 `DDPOptimizer`，默认 True;
   * `cache_size_limit`: TorchDynamo 会把编译好的函数被保存在 frame 的 cache 中，从而避免再次编译相同的函数和输入。默认情况下 cache_size_limit 大小为 64，也就是说，对于同一个 Python 函数，它的输入最多可以有 64 种变化，超过这个限制后 TorchDynamo 不再编译该函数。
  
3. `**kwargs`, 即 `torch.compile()` 接口所需参数，详细可参照 [pytorch 接口文档](https://pytorch.org/docs/2.0/generated/torch.compile.html?highlight=torch+compile#torch.compile)，这里介绍下几个主要参数：
   * `fullgraph`: 是否允许在 compile 时把生成的 dynamo graph 拆分成若干个子图，默认 False;
   * `dynamic`: 是否要支持动态 shape tracing，默认 False;
   * `backend`: 编译 backend，可选 [`inductor`, `cudagraphs`, `onnxrt`, `tvm`, `nvprims_nvfuser`], 默认 "inductor"，;
   * `mode`: 编译模式，可选 [`default`, `reduce-overhead`, `max-autotune`]，优化效果依次提升，但编译耗时也依次增加;


> Note：特别说明，HAT 中 `TorchCompile` 接口 {ref}`torchdynamo/how_to_use_torchdynamo_speedup_inference` 主要用于 TensorRT (torch<2.0) 的推理加速。若要训练加速，且 torch>=2.0, 请使用 `Torch2Compile` 接口.

更多 `torch.compile` 设置细节，可查看文档中的 Pytorch 链接，若不知如何设置，保持默认设置即可。

## FAQ:

### 1. 为什么使用 torch compile 后，训练速度变慢很多？如何 debug ？

首先，如果是多卡 DDP 训练，那么 DDP 上 `torch.compile` 的加速效果可能会打折扣，具体原因可见 [pytorch discuss](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)，所以可以先尝试在单机单卡上验证 `torch.compile` 的加速效果。

接着，如果单卡上，使用 `torch.compile` 后速度仍然变慢很多，那就是有问题，需要 debug 分析了

> 思考：torch compile 引入的额外耗时主要是编译上，因此训练速度变慢，很大的因素就是做了太多次编译，甚至可能在一边跑一边编译，而会造成编译多次的原因可能是：
> 1. 模型中有太多分支（比如 if-else），在首次执行某一分支时，都会做一次编译；
> 2. 模型中存在 dynamic shape 的 Tensor，当执行到某个 dynamic tensor，而 torch 已编译缓存中又没有当前 shape 的缓存时，也会执行重新编译的操作；
>
> 针对上述两种情况：
> 1. 分支太多：因为出现 if-else 的时候，torch dynamo 会有相应的 graph-break，所以，可以通过 graph-break 信息来辅助确认模型中哪个 Module 分支过多；
> 2. dynamic shape：这种情况会是在相同代码行处做 re-compile，所以可以通过获取 re-compile 次数信息来定位；

基于上述两点，HAT 中开发了相应的 debug 工具，使用方式如下：

```python
float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=[],
    # 添加 Torch2Compile，使用 torch.compile()
    compiler=dict(
        type="Torch2Compile",
        ...
        backend="inductor",
        ...
    ),
    data_loader=...,
    optimizer=...,
    batch_processor=...,
    # 修改: 添加 DynamoProfiler
    profiler=dict(
        type="DynamoProfiler",
        output_dir="./tmp_profile_outputs",
        filename="dynamo_result"       
    )
    ...
)
```

然后执行训练（几个 step 就行），会生成 `graph-break` 和 `re-compile` 信息的 summary 文件。

注意：
* `graph-break` 信息: 该信息只是用来辅助查看 `if-else` 分支情况，并不是意味着每次 `graph-break` 都额外增加了编译耗时；
* `re-compile` 信息: 部分模块可能因其复用性太高，在统计时会显示 re-compile 次数很多(比如 `ConvModule2d`)，可以暂时忽略，或尝试增大 `cache_size_limit` 试试；

针对 `graph-break` 和 `re-compile` 都明显出现次数较多的 Module，可以尝试将其 `skip_compile` 再验证效果。

**Tips：** 
* torch compile 结合 AMP 使用，加速效果更佳
* 会导致 graph-break 的语法及原因可见 [torch 文档](https://pytorch.org/docs/stable/torch.compiler_faq.html#limitations)


### 2. 为什么在 Trainer 中可以使用 torch compile，而在 DDP Trainer (DistributedDataParallelTrainer) 中报错了？

依据 [Pytorch 文档](https://pytorch.org/docs/stable/notes/ddp.html#example) ，推荐在 ddp_wrapper 之后使用 `torch.compile`，即：
```python
ddp_model = DDP(model, device_ids=[rank])
ddp_model = torch.compile(ddp_model)
```
在 [Pytorch Discuss](https://discuss.pytorch.org/t/how-should-i-use-torch-compile-properly/179021) 中提到 `torch.compile` 在 ddp_wrapper 后使用时，torch compile 会对 DDP 分布式中的通信做优化，但当前可能还未完美支持，可能会出现一些错误，并建议遇到错误时尝试在 ddp_wrapper 之前做 compile，即：
```
model = torch.compile(model)
ddp_model = DDP(model, device_ids=[rank])
```
所以，当 Trainer 中可以使用 torch compile 而 DDP 中报错时，可以先尝试把 compile 放在 DDP 之前，HAT 中 config 修改如下：
```python
float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    # 修改1: 把 compiler 设置为 None
    compiler=None,
    model_convert_pipeline=[
        ...
        # 修改2：把原本 compiler 的参数，放到 model_convert_pipeline 中
        dict(
            type="Torch2Compile",
            skip_modules=[".*loss.*"],
            regex=True,
            ...
        ),
    ],
    data_loader=...,
    optimizer=...,
    batch_processor=...,
    ...
)
```

如上述修改之后仍然出现错误，可以到社区查找是否有解决方案，或及时联系 HAT 团队。

更多 Compile 使用问题可见 [TORCH COMPILE FAQ](https://pytorch.org/docs/stable/torch.compiler_faq.html) 和 [PYTORCH 2.0 TROUBLESHOOTING](https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html)