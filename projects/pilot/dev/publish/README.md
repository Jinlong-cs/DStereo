# 模型编译

## 本地

在本地trace并编译模型(编译部分可在本地或aidi执行，通过`COMPILE_MODE`参数指定)，使用时请注意本地环境。

```shell
# cd hat root
python3 projects/pilot/dev/publish/trace_compile_model.py |
   --sub-project ${SUB_PROJECT} |
   --publish-version ${PUBLISH_VERSION} |
   --compile-mode ${COMPILE_MODE} (--overwrite)
```

其中`PUBLISH_VERSION`参数在`COMPILE_MODE = "local"`的情况下，只与存下来的各种文件名相关。而在aidi编译模式下，才会成为上传至模型管理平台的模型版本号。

## 集群

这种模式下，编译只允许提交到aidi进行。

```shell
# cd hat root
python3 projects/pilot/tools/publish/submit_trace_compile_model.py |
   --sub-project ${SUB_PROJECT} |
   --publish-version ${PUBLISH_VERSION} |
   --project-id ${project_id} |
   --current-cluster ${current_cluster} |
   --compile-mode local |  # local or aidi; 是否aidi云端编译
   --overwrite
   (--release)  # 是否发布模式，决定模型发布名等
```

这种情况下，`PUBLISH_VERSION`应该是由字母v开头，3个数字用`.`分割的版本号。

## 运行流程

无论在本地还是集群，执行的内容都如下所示：

1. int_infer模型生成：
   调用`check_and_convert`方法
   - 根据发布config中每个模型`update_cfg`字段所声明的阈值，更新config里的desc信息。
   - 根据模型checkpoint的参数信息，以及对应的模型cfg中所定义的模型结构，通过trace的方式得到int_infer模型。

2. 编译得到hbm上板模型
   1. 本地模式（`COMPILE_MODE = "local"`）：执行`compile_local`方法，轮询`pub_cfg`中定义的各模型设置，并按照读取得到的相应参数，编译对应的int_infer模型。最后调用`hbdk-pack`命令将编译好的所有模型打包。
   2. 艾迪模式（`COMPILE_MODE = "aidi"`）：
      1. 执行`upload_to_aidiexp`方法，轮询`pub_cfg`中定义的各模型设置，将上一步trace得到的int_infer模型通过`aidisdk`上传至艾迪实验模型管理平台中，并记录相关信息。
      2. 执行`compile_publish_aidi`方法，轮询`pub_cfg`中定义的各模型设置，对每个待编译模型，输入其对应实验模型信息，并按照`pub_cfg`中的相应内容设置参数。最后提交编译任务，等待编译完成。

## Q & A

Q. config存放在哪？如何调用？
A: 存放在[fillback_eval](../../dev/publish/cfg)中，并通过pub_cfg_{sub-project}.py调用。


# 模型推理结果输出

```shell
# cd projects/pilot/dev/publish
python3 projects/pilot/dev/publish/get_result.py --sub-project ${SUB_PROJECT} --pub-version ${PUBLISH_VERSION}
```

其中`SUB_PROJECT`可以按需指定。而`PUBLISH_VERSION`指定与否，只会影响dump出结果的命名。

运行完成后，结果会被存在一个根目录以`{SUB_PROJECT}_{model_version}.tar.gz`命名的文件中。随后将此文件提供给下游即可。

# 手动触发编译和一致性对齐任务

非Tag发版情况下，手动触发编译和一致性比对流程。

```shell
# cd hat root
python3 projects/pilot/tools/publish/submit_publish_model.py |
   --sub-project ${SUB_PROJECT} |
   --publish-version ${PUBLISH_VERSION} |
   --project-id ${project_id} |
   --current-cluster ${current_cluster}
```


# 拆分模型int_infer一致性校验

```shell
# cd hat root
python3 projects/pilot/dev/publish/verify_bev_split.py --model-address ${MODEL_ADDRESS} --cfg-path ${CONFIG_DIR} --model-setting ${MODEL_SETTING}
```

本地对比int模型输出结果。报错则说明bev的int模型在拆分前后输出结果不一致。
