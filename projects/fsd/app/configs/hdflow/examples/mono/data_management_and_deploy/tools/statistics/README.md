# 部署数据数据统计工具
------
## OverView
这里为Mono 的同学提供数据统计工具。目前支持三种类型数据源：

- leaderboard id
- densebox rec
- densebox json


# Getting Started
## Install
- 请根据hdflow官方配置完成安装。

## 运行统计脚本
1. 参考示例的[配置文件](configs/statistic_sources.py), 设置数据源信息。每个统计任务由`StatisticItem`定义。如下是一个示例：
```python
StatisticItem(
    source_type=StatisticSource.leaderboard_id,
    values=[6042257],
    item_name="test_with_6042257",
    collect_keys=[
        # attrs
        "attrs.tags.time",
        "attrs.tags.weather",
        "attrs.tags.sensor",
        "attrs.tags.fov",
        "attrs.tags.isp",
        "attrs.tags.scene",
        # class types
        "vehicle.attrs.Orientation",
        "vehicle.attrs.confidence",
        "vehicle.attrs.ignore",
        "vehicle.attrs.occlusion",
        "vehicle.attrs.type",
    ]
),
# source_type: 数据源的类型。
# values： 数据源的值。这里values可以是多个 rec、多个 leaderboard id 等等。
# item_name: 统计任务的名字，它可能对应一个独立的路径保存结果。
# collect_keys：需要统计的的量。
```
```python
# 关于 collect_keys的规则.
# 1. 常规的字典嵌套，比如
input = dict(a=dict(b=dict(c=123)))
# 可以通过 a.b.c 统计到 c中的内容
# 2. 嵌套中可以出现 list，比如
input = dict(a=[
    dict(b=dict(c=111)),
    dict(b=dict(c=222)),
    dict(b=dict(c=333)),
    ]
)
# 我们依然可以通过 a.b.c 统计到 c中的内容，区别在于这条input会为a.b.c提供三条内容
```

2. 完成配置后，执行下面的命令，运行统计
```shell
cd examples
export PYTHONPATH=$(pwd)
config_path=mono/data_management_and_deploy/tools/statistics/configs/statistic_sources.py
result_dir=./
python3 mono/data_management_and_deploy/tools/statistics/statistic_data.py --statistic-config $config_path  --statistic-result-dir $result_dir
```
3. 运行完成后，结果会保存在 `statistic_result_dir/item_name`中。
