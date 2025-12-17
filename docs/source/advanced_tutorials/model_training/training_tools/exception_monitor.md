# 如何使用异常值监控功能

在模型训练过程中，可以使用异常值监控功能来对训练过程中产生的数据进行监控，帮助及早发现训练过程中的异常情况。

## 使用方式

开启异常值监控的功能，需要依赖 `metrics_updater`，其中 `metrics_updater` 作为 metrics 生产者， `exception_monitor` 读取 metrics 信息进行判断，输出异常值的信息进行warning并可传输至飞书信息，以便及早发现。HAT 里用 `MetricUpdater` 和 `ExceptionMonitor` callbacks实现此功能，使用方式如下(config 文件中配置):

```python

metric_updater = dict(  # 常用的metrics_updater的配置
    type="MetricUpdater",
    metric_update_func=update_metric,  
    step_log_freq=1,
    epoch_log_freq=1,
    log_prefix=task_name,
    step_storage_freq=1,  # metrics信息存储的 step 频率
    epoch_storage_freq=1,  # metrics信息存储的 epoch 频率
    storage_key="loss_monitor_obj",  # metric_updater执行后会将metrics信息按照设置的key值暂存起来
)
# 设置 ExceptionMonitor callback
exception_monitor_callback = dict(
    type="ExceptionMonitor",
    step_monitor_freq=1,  # metrics信息监控的 step 频率
    epoch_monitor_freq=1,  # metrics信息监控的 epoch 频率
    monitor_prefix="ToyTask",
    monitor_func=default_monitor_func,  # 判断metrics信息异常的函数，提供默认值
    monitor_key="loss_monitor_obj",  # 按照key值读取上述metric_updater暂存的信息
)
# 提供判断metrics信息异常的默认函数，即出现 nan 或 inf，同时支持用户配置
def default_monitor_func(values: List):
    exception_values = []
    for k, v in values:
        try:
            if math.isnan(v) or math.isinf(v):
                exception_values.append((k, v))
        except Exception:
            pass

    return exception_values

# 注意: 开启异常值监控，在trainer中配置callback时，exception_monitor_callback 应该在 metric_updater 后面
xxx_trainer = dict(
    ...
    callbacks=[
        # the order of callbacks affects the logging order
        ...
        metric_updater,  # 先配置 metric_updater callback
        exception_monitor_callback, # exception_monitor_callback 放在 metric_updater 后面
        ...
    ],
    ...
)

```

之后启动训练即可，训练过程中产生的 exception 异常信息会同步打印进行warning并可发送至飞书客户端，以便及早发现。
