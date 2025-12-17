# 回灌评测链路使用说明文档
## 使用前注意
回灌以及回归是在板端执行，需要提前申请板端资源，如IDC-NewAge --> 嵌入式设备集群 --> project-j5-pilot50-badcase-idc-newage


## 本地执行
确保本地hdflow满足[requirement](../../requirements.txt)要求。BEV模型回归和回灌要求horizon-hdflow版本 >= 0.8.4b202312210650+788a69a

切换至hat根目录，运行example：

```python
python3 projects/pilot/tools/fillback_eval/pipeline.py  --sub-project test \
                                                        --version v0.0.1 \
                                                        --fillback-type evs \
                                                        --enable-report-diff \
                                                        --fillback-cluster project-j5-pilot50-badcase-idc-newage \
```

参数说明：

- `--sub-project`: 项目名；链路会通过该名称寻找对应config与datasets。
- `--fillback-type`: 回灌评测类型；目前支持 evs: 回灌评测，issue: 回归评测。
- `--version`: 版本号；若指定，会覆盖config对应配置。
- `--app`: 软件包；[软件包管理平台](http://aidi.hobot.cc/package/package-management)上存放的软件包，通过名称指定。
- `--update-hbm`: 若指定，软件包会将内部hbm更新为该hbm文件；支持[aidi publish model](http://model.aidi.hobot.cc/models/publishedModel?range=mine&current=1)路径，例如[`MCP5.0_galaxy_x3c_pe:v14.5.5`](http://model.aidi.hobot.cc/models/publishedModel/43779?activeKey=3)，或bucket路径指定。BEV模型回灌和回归，若更新hbm，需提供模型包如 http://gallery.hobot.cc/download/auto/pilot_algo/pilot5_1_model_release/byd_ek/project/snapshot/linux/arm/general/basic/pilot_perception_v1.1_ek_20231121-223310/byd_ek-pilot_perception_v1.1_ek_20231121-223310.zip。
- `--disable-eval`: 关闭评测。
- `--enable-report-diff`: 开启评测report比对。运行baseline任务时，不建议加上--enable-report-diff选项；运行update任务，如果用户想要开启--enable-report-diff功能，则需要在相应的issue dataset里面加上issue_job_compared，issue_job_compared从该页面获取（http://issue.aidi.hobot.cc/task/list?searchQuery=%7B%22range%22%3A%22all%22%7D）。
- `--fillback-cluster`: 用户传入的嵌入式设备回灌队列，非必须，用户没有传参，则传入configs目录中对应的项目config的默认嵌入式回灌队列。


## 集群执行

切换至hat根目录，运行example:
```python
python3 projects/pilot/tools/fillback_eval/submit_cluster.py  
                                                --sub-project test \
                                                --version v0.0.1 \
                                                --fillback-type evs \
                                                --project-id PDT20220001 \
                                                --current-cluster svc-aip-cpu \
                                                --fillback-cluster project-j5-pilot50-badcase-idc-newage \
                                                --enable-report-diff \
                                                
```

相比本地执行，仅需增加
- `--project-id`: 项目号。
- `--current-cluster`: 执行集群；使用cpu集群即可。
- `--fillback-cluster`: 用户传入的嵌入式设备回灌队列，非必须，用户没有传参，则传入configs目录中对应的项目config的默认嵌入式回灌队列。


需要注意，默认使用[pilot_cluster_cfg](../pilot_cluster_cfg.yaml)中fillback_eval的cpu docker

## Q & A

Q. config存放在哪？如何调用？
A: 存放在[fillback_eval](../fillback_eval/configs)中，并通过{sub-project}.py调用。
在config中，分别通过evs、issue来指定不同评测类型使用的配置。
类似地，[datasets](../fillback_eval/datasets)通过{sub-project}_{fillback-type}_datasets.py来调用。

Q. 如何针对不同数据集，对提交job时的参数进行一些修改？
A: 我们对数据集提供了meta参数，用户可以将需要的参数信息放在其中，并自定义配置config来处理datasets中的meta参数。
[参考样例](../fillback_eval/configs/project_utils.py)。

Q. 开启report diff功能后，在哪能找到diff_report？
A: 会保存到两个路径：本地(集群保存到output中) 以及 bucket(dmpv2://matrix2/users/{user_name}/fillback_eval_xlsx)上