## 辅助脚本

### 注意事项

在运行下列工具脚本前，需要先设置环境变量，方式如下:

```
source set_environment.sh
```

### 数据采集相关

说明: 基于脚本1进行格式校验，确认无格式错误后，再运行脚本2进行数据生成。

1. 数据采集文档格式转换及格式检查脚本.

```
环境： python 3 , pandas
使用方法查询: python raw_samples_2_std_json.py --help
说明: 数据采集文档路需保存为csv格式。
数据采集文档地址: https://horizonrobotics.feishu.cn/wiki/wikcnEpv68cNRakCxwbBNUUSEqd?sheet=a08665
```

2. 基于转换后的标准json数据生成nlu数据集.

```
环境： python 3
使用方法查询：python std_json_2_nlu_format.py --help
说明: 需配置输入路径以及正样本/负样本存储路径， 输入路径为 raw_samples_2_std_json.py 脚本生成的标准格式json文件。
```

### 其他

1. 数据集格式转换脚本

```
环境： python 3
使用方法查询: python model_testset_2_nlu_testset.py --help
说明: 将nlu模型测试集格式转为nlu引擎测试集格式。
```

2. 有效数据过滤脚本

```
环境： python 3
使用方法查询: python filt_valid_dataset.py --help
说明: 基于nlu_dataset数据校验逻辑,统计有效数据，过滤无效的数据样本。
用途举例：
a)新增测试集，用此脚本校验合法数据比例是否符合预期。
b)进行板端一致性验证前，先基于此脚本过滤数据集，再使用model_testset_2_nlu_testset.py脚本转换数据集格式，保证模型测试集与板端测试集有效样本数量完全一致。
```
