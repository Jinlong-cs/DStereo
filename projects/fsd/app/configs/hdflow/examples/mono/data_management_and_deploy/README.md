# Mono数据部管理、部署工具
------
## OverView
```text
      量产环境中，标注好的数据可能会用作训练(数据打包)、评测(数据创建测试集)或其他用途(统称为数据部署)。
    数据在转化为新的结构后，需要对源数据做到可跟踪。这也是数据部署工具需要解决的问题。
      在Mono项目中，目前需要解决的是训练、评测数据集的规范化管理。对于训练，需要通过用户给定的一组数据集，
    完成数据打包；对于评测，则创建leaderboard评测集。
```

# Install

- 请根据hdflow官方配置完成安装。
- 有些任务在部署时需要刷库处理(e.g. parsing"车底相连")，这是注意需要安装下开源第三方库。

# 使用文档

用户可以根据batch_runner.py中提供的运行示例跑通流程，阅读相关实现,注意batch_runner.py需要在hdflow/examples下运行。


详细的使用文档，请参考飞书文档:

[[Mono]数据部署(打包/创建评测集)迁移Hdflow ](https://horizonrobotics.feishu.cn/docs/doccnHUGTpezYjdIBZcAwVri9Ze#l0X23Q)

