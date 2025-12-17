执行脚本
========

在`HAT`中,用户可以直接使用和修改的功能主要包含tools和configs两部分的内容。其中tools主要是核心的功能模块,包含训练验证可视化等。而configs主要是包含功能模块执行过程中可以配置的选项和参数。

这篇教程告诉大家tools主要包含哪些核心功能,以及config的开发规范和使用流程。

大部分情况下tools工具的执行模式都是需要输入一个config,除了部分与数据集或者单张图可视化相关的工具。因此通用的执行范式可以总结成这样：

    .. code-block:: bash

       python3 tools/${TOOLS} --config configs/${CONFIGS}

这里主要介绍一下tools的核心功能和对外接口。


tools
---------

目前的tools一共有7个python脚本,每个脚本具有不同的功能。

train.py是训练工具,常用的参数为:

    --step {float, float_freeze_bn, qat, int_infer}: 不同的训练和预测阶段

    --config CONFIG, -c CONFIG: config文件路径

    --device-ids DEVICE_IDS, -ids DEVICE_IDS 运行的GPU列表

    --dist-url DIST_URL: 多机运行指定的server地址,默认是auto

    --launcher {mpi}: 多机运行的启动方式,目前只支持mpi

    --pipeline-test: 是否运行pipeline测试

    --export-ckpt-only: 只导出checkpoint

    --val-only: 只运行val

    --val-ckpt: val时候的checkpoint路径

align_bpu_validation.py是精度验证工具,提供和上板完全对齐的结果。常用参数列表为:

    --config CONFIG, -c CONFIG: config文件路径

calops.py是网络计算量统计工具,常用参数列别为:

    --config CONFIG, -c CONFIG: config文件路径

    --input-shape: 输入shape

compile_perf.py是编译和perf工具,常用参数列表分别为:

    --config CONFIG, -c CONFIG: config文件目录

    --out-dir OUT_DIR: 输出目录

    --opt {0,1,2,3}: 编译时的优化选项

    --jobs JOBS: 编译时的线程数

compile_standalone.py是编译工具,可以接受pt文件。常用的参数列表为:

  --input-size INPUT_SIZE: 输入shape

  --name NAME: hbm的名字
  --opt OPT {O0, O1, O2, O3}: 编译器的优化级别

  --march MARCH: 编译对应的芯片平台

  --skip-compile: 是否跳过编译，只做perf

infer.py是用来做单图预测的工具。常用的参数列表为:

    --step {float, float_freeze_bn, qat, int_infer}: 不同的训练和预测阶段

    --config CONFIG, -c CONFIG: config文件路径

    --img IMG: 用来做infer的图片

    --model: 预训练的模型地址

    --visual-type: 可视化的方式

datasets目录是数据集相关的打包和可视化工具。
