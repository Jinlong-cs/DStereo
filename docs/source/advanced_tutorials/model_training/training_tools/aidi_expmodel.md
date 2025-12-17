# 如何使用 AIDI 实验管理功能

## 模型管理

HAT `AIDIExperimentManager callback` 实现了对接艾迪平台中实验管理功能，当训练完成后，模型将自动化上传至艾迪平台中的实验管理模块。
> AIDI 平台目前是通过 Artifact 实现对模型的管理。

### 使用方式

通过在 `config` 文件中增加 `AIDIExperimentManager callback` 即可：

```python
aidi_expmodel_callback = dict(
    type="AIDIExperimentManager",
    model_name=model_name,  # 指定模型名称
    model_version="v0.0.1",  # 指定模型版本，默认 None
    save_model='last',
    upload_progressive_checkpoint=False,  # 默认 False，则只会上传一个 Loop end 后产生的 checkpoint.
    overwrite_file=False,  # 是否对已上传文件进行覆盖，默认 False
)

float_trainer = dict(
    ...
    callbacks = [
        ...
        aidi_expmodel_callback, # 增加 AIDIExperimentManager callback
    ]
)

```
其中，`AIDIExperimentManager` 参数代表含义如下：
- model_name:（`str`,必选）实验模型目录名称。
- save_model：（`str`,必选）保存模型的类型。可选参数：`last`或`best`。
- model_version：(`str`, 可选) 模型版本号。
- upload_progressive_checkpoint: (`bool`, 可选) 是否上传训练过程中 `by_epoch` 或 `by_step` 产生的 checkpoint 文件。
- overwrite_file: (`bool`, 可选) 是否对已上传文件进行覆盖。如果 False，实验管理中会保留多个 Artifact(即多个 checkpoint 文件记录)，如果 True，则只保留一个 Artifact(即只保留一个 checkpoint 文件记录)。

结合示例对 `upload_progressive_checkpoint`、`overwrite_file` 的设置进一步说明：
* 默认设置(`upload_progressive_checkpoint=False` && `overwrite_file=False`): 该设置下仅会在训练结束时，保存当前训练 `stage` 在 `loop_end` 产生的模型文件，且只有一个，比如[示例一](http://model.aidi.hobot.cc/experiment/140/run/173):
* `upload_progressive_checkpoint=True` && `overwrite_file=False`: 该设置下会保存当前训练 `stage` 在训练过程中产生的所有模型文件，且 AIDI 实验管理系统会有多条记录，比如[示例二](http://model.aidi.hobot.cc/experiment/138/run/171):
  
* `upload_progressive_checkpoint=True` && `overwrite_file=True`: 该设置下也会保存当前训练 `stage` 在训练过程中产生的模型文件，但会以覆盖的形式，在 AIDI 实验管理中只会存在一条记录，比如[示例三](http://model.aidi.hobot.cc/experiment/139/run/172):

除了上述 `config` 配置之外，训练启动命令如下：
* 开发机上需要用 `aidiexp_run.py` 来启动 `train.py`，并添加 `--experiment-name`(必须)、`--project-id`(必须)、`--run-name`(可选)、`--experiment-path`(可选) 等参数，例如:
    ```bash

    SCRIPT="tools/train.py"  # 指定要执行的脚本（tools/train.py）
    SCRIPT_ARGS="--config xxx/xxx_config.py --stage" # 传给执行脚本（tools/train.py）的参数

    python3 tools/aidiexp_run.py \
        --experiment-name "test_exp" \
        --run-name "test_run" \
        --project-id "xxx" \
        --script ${SCRIPT} \  # 指定 script
        --script-args ${SCRIPT_ARGS}
    ```

    其中，
    * `--script`: 为要执行的脚本，例如 `tools/train.py`;
    * `--script-args`: 为执行脚本(例如 `tools/train.py`) 所需的参数;


- 提交集群时，需要给`submit.py` 传入 `--experiment-name`(必须)、`--run-name`(可选) 参数:
    ```bash
    python3 submit.py --config k8s_config.py --cluster share-3090-idc --experiment-name "test_exp" --run-name "test_run"
    ```

启动训练后，可在艾迪平台[实验管理](http://model.aidi.hobot.cc/experiment)页面看到相关实验信息。


## AIDI Model Tracking

除了上面 "模型管理" 功能之外，AIDI 实验管理还支持 `AIDI Model Tracking` 功能，该功能是一套把model metrics tracking上报到web模型训练链路展示的模型生命周期管理的工具，目的是实现模型的全生命周期管理，做到模型的可追溯、可复现、可重试，并辅助进行任务的运营统计。

HAT 目前支持训练任务、评测(AIDIEval)任务过程中的信息上报。

### 训练任务: config 配置、训练 exception 异常、metrics 信息上报

开启这项功能后，在训练任务中使用的config 配置，以及训练失败后产生的 exception 异常、trackback、训练过程中的metrics中的loss值等信息均会上报到实验管理。

训练任务信息上报也通过 `AIDIExperimentManager callback` 实现，因此要使用训练任务配置、训练任务异常信息上报功能，`config` 文件中设置了 `AIDIExperimentManager callback` 即可。

此外，要开启metrics上报，需要依赖 `metrics_updater`，其中 `metrics_updater` 作为上游 metrics 生产者， `tracking` 作为下游 metrics的消费者逐步消费并进行上传。

```python
enable_model_tracking = True # 为 False 则关闭对config、exception、metric 等信息的上报，默认为False

metric_updater = dict(  # 常用的metrics_updater的配置
    type="MetricUpdater",
    metric_update_func=update_metric,  # metrics update执行后会将metrics信息push给 aidi_expmodel_callback callback
    step_log_freq=10 if not pipeline_test else 1,
    epoch_log_freq=1,
    log_prefix=task_name,
)
# 设置 AIDIExperimentManager callback (和上面 "模型管理" 部分的设置保持一致即可)
aidi_expmodel_callback = dict(
    type="AIDIExperimentManager",
    model_name=model_name,
    save_model='last',
    upload_progressive_checkpoint=False,
    overwrite_file=False, 
)

# 注意: 开启 tracking 后，在trainer中配置callback时，aidi_expmodel_callback 应该在 metric_updater callback 和 ckpt_callback 后面
float_trainer = dict(
    ......
    callbacks=[
        # the order of callbacks affects the logging order
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            warmup_by="epoch",
            warmup_len=0,
            lr_decay_id=[15, 25],
            step_log_interval=10,
        ),
        metric_updater,  # 先配置 metric_updater callback
        ckpt_callback,  # 先配置 ckpt_callback
        val_callback,
        aidi_expmodel_callback, #  aidi_expmodel_callback callback 放在 metric_updater callback 和 ckpt_callback 后面
    ],
    ......
)

```


注意:
* 如果是在开发机上运行，还需传给 `aidiexp_run.py` 脚本 `--enable-tracking` 参数:
    ```bash

    SCRIPT="tools/train.py"  # 指定要执行的脚本（tools/train.py）
    SCRIPT_ARGS="--config xxx/xxx_config.py --stage" # 传给执行脚本（tools/train.py）的参数

    python3 tools/aidiexp_run.py \
        --experiment-name "test_exp" \
        --run-name "test_run" \
        --project-id "xxx" \
        --enable-tracking \  # 允许 tracking
        --script ${SCRIPT} \  
        --script-args ${SCRIPT_ARGS}
    ```

* 同理，如果是提交到集群的job，也需要给 `submit.py` 脚本传入 `--experiment-name`(必须)、`--run-name`(可选) 的基础上，增加 `--enable-tracking` 参数:
    ```bash
    python3 submit.py --config k8s_config.py --cluster share-3090-idc --experiment-name "test_exp" --enable-tracking

    ```


### 预测任务： 评测 prediction 信息上报

预测的 tracking 功能是为了把模型、预测job、评测结果关联到一起。类似于训练过程中的 `tracking` 功能，在评测过程中，也可以开启 `tracking`，将评测信息进行上报，`config` 配置如下：

```python

enable_model_tracking = True # 1. 设置参数，开启 tracking 上报

#  2. HAT 评测用到的 AIDIEval callback，开启 tracking 时不需要对原有 callback 做任何更改
aidi_eval_callback = dict(  
    type="AIDIEval",
    aidi_eval_dataset_id=...,
    output_root=...,
    prediction_name=...,
    ...
)

# 3. 开启 tracking 后，在 LoadCheckpoint 设置 AIDI checkpoint 路径和 enable_tracking 即可.
float_predictor = dict(
    ......
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        ...
        converters=[
            dict(
                type="LoadCheckpoint",
                # 3.1 设置 AIDI 实验管理中的 checkpoint 路径，根据训练过程中 AIDIExperimentManager 产生，
                # 其中, artifact_alias 有以下几种设置方式（一般使用时设置 "latest" 即可）：
                # - "latest": 使用最新版本.
                # - "last" 或 "best": 也是使用最新版本（根据训练时 `AIDIExpModel callback`设置 `save_model` 是 "last" 还是 "best" 决定）.
                # - f"{training_stage}-{step/epoch}_x": 训练过程中产生的版本（当训练时设置了 upload_progressive_checkpoint=True && overwrite_file=False 时才会有）.
                # - "v0.0.1": 训练时设置的 model_version.
                checkpoint_path=f"aidi_artifact://{model_name}/{training_stage}/{artifact_alias}/{checkpoint_name}",
                enable_tracking=enable_model_tracking,
            ),
        ],
    ),
    callbacks=[
        ...,
        [   ...,
            aidi_eval_callback,   # 4. 设置 AIDIEval callback
            ...,
        ],
        ...,
    ],
   
)

```
注：这里AIDIEval callback只是示例， 具体使用方法可见 {ref}`aidi_eval/how_to_use_aidi_eval`。

注意:
* 同训练时一样，如果是在开发机上运行，需要用 `aidiexp_run.py` 来启动 `predict.py`，并添加 `--experiment-name`(必须)、`--run-name`(可选)、 `--project-id`(必须)、 `--enable-tracking` 参数:

    ```bash

    SCRIPT="tools/predict.py"  # 指定要执行的脚本（tools/predict.py）
    SCRIPT_ARGS="--config xxx/xxx_config.py --stage" # 传给执行脚本（tools/predict.py）的参数

    python3 tools/aidiexp_run.py \
        --experiment-name "test_exp" \
        --run-name "test_run" \
        --project-id "xxx" \
        --enable-tracking \  # 允许 tracking
        --script ${SCRIPT} \  
        --script-args ${SCRIPT_ARGS}
    ```

* 同理，如果是提交到集群的job，也需要给 `submit.py` 脚本传入 `--experiment-name`(必须) 、 `--run-name`(可选)、`--enable-tracking` 参数:
    ```bash
    python3 submit.py --config k8s_config.py --cluster share-3090-idc --experiment-name "test_exp" --run-name "test_run" --enable-tracking

    ```

按上述配置之后，正常启动 predict 脚本即可，程序运行结束后，可在艾迪平台实验管理页面看到模型、预测job、评测结果已关联到一起，例如 [AIDI模型-评测结果关联示例](http://model.aidi.hobot.cc/experiment/139/run/187)


### 编译任务： 模型编译信息上报

现在 HAT 中也支持了模型编译(`compile_standalone.py` 和 `pack_hbm.py`)信息、产物的上报。使用方式与训练、评测中类似。不过需要注意的是，不同于训练和评测，不论在开发机还是在集群，要对 `compile_standalone.py` 和 `pack_hbm.py` 信息上报时，这两个脚本都必须通过 `aidiexp_run.py` 的方式来执行。例如：

* 在开发机上
    ```bash

    SCRIPT="tools/deploy/compile_standalone.py"  # 指定要执行的脚本
    SCRIPT_ARGS="xxx" # 传给执行脚本的参数

    python3 tools/aidiexp_run.py \
        --experiment-name "test_exp" \
        --run-name "test_run" \
        --project-id "xxx" \
        --enable-tracking \  # 允许 tracking
        --script ${SCRIPT} \  
        --script-args ${SCRIPT_ARGS}
    ```

* 提交集群任务时：

    ```bash

    SCRIPT="tools/deploy/compile_standalone.py"  # 指定要执行的脚本
    SCRIPT_ARGS="xxx" # 传给执行脚本的参数

    python3 tools/aidiexp_run.py \
        # 由于 submit.py 中已经设置了 AIDI 实验相关信息，这里不需再传入
        #--experiment-name "test_exp" \
        #--run-name "test_run" \
        #--project-id "xxx" \
        #--enable-tracking \  # 允许 tracking
        --script ${SCRIPT} \  
        --script-args ${SCRIPT_ARGS}
    ```

### 上报完成后的数据展示和可视化

上报完成后，在 aidi 平台的[实验管理](http://model.aidi.hobot.cc/experiment)页面，可以查看到一个模型的训练链路各个节点信息，具体详情可以咨询aidi。


### FAQ

**Q1. 上报是否对训练任务有其他影响？**

**A:**  这些上报都是异步的形式，且使用很少量的网络IO，对训练任务的影响微乎其微。即使是都失败了，也不会对任务有影响。
