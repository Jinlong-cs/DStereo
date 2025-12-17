# 在 detectron2 中使用 HAT DenseboxDataset

HAT 提供了兼容接口，可在社区框架 detectron2 中调用 HAT 中的 `DenseboxDataset`。


## 安装公版 detectron2

当前支持的 detectron2 为 v0.6，所需 cuda 与 torch 版本与 HAT 相同，具体为：


|   CUDA   |   torch1.10   |    torch1.9   |
| :------- | :----------:  | :----------:  |
|   11.1   |     支持      |     支持      |
|   10.2   |     支持      |     支持      |
|   10.1   |      --       |      --       |



支持两种方式安装:
    
1. pip 安装:
   ```bash
   pip3 install --user detectron2 -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/detectron2/cu102/torch1100 --trusted-host art-internal.hobot.cc
   ```
    安装不同 cuda 和 torch 版本的包，只需要更改为相应的版本即可。详细安装请见 [detectron2安装](https://horizonrobotics.feishu.cn/wiki/wikcnUtqpbxQ1qIx5WNu1mpd1Gg#Bhm6XD)
  
    HAT 和 Horizon_plugin_pytorch 的安装请见 HAT 文档 `QUICK START/安装` 部分。

2. wheel 包本地安装
   detectron2 官方 wheel 包已经存放到 HDFS 中，路径格式为：
   ```bash
   hdfs://hobot-bigdata/user/mengyang.duan/hat_openlab/open_detectron2/wheels/${torch_version}/${cuda_version}/detectron2-${d2_version}+${cuda_version}-cp36-cp36m-linux_x86_64.whl
   
   # 其中，
   #    torch_version 的值是  "torch1.10" 或 "torch1.9"，与上表中对应
   #    cuda_version  的值是  "cu111" 或 "cu102"，与上表中对应
   #    d2_version    的值是  "0.6"
   ```
   例如：
   ```bash
    hdfs dfs -get hdfs://hobot-bigdata/user/mengyang.duan/hat_openlab/open_detectron2/wheels/torch1.10/cu111/detectron2-0.6+cu111-cp36-cp36m-linux_x86_64.whl
    pip3 install ./detectron2-0.6+cu111-cp36-cp36m-linux_x86_64.whl
   ```

## 在 detectron2 中使用 HAT DenseboxDataset

1. 修改 detectron2 的 config 文件
   
    由于 detectron2 CfgNode 的限制，dataset 信息的设置需要借助辅助文件。例如：
    - 原本 detectron2 DATASETS 的设置：
        ```python
        DATASETS:
          TRAIN: ("coco_train_2017", )
          TEST: ("coco_test_2017",)
        ```
    - 在使用 HAT 里的 DenseboxDataset 时，设置:
        ```python
        DATASETS:
          TRAIN: ({"rec_dataset_path_file": "/xxx/dataset.yaml", "task_name": "vehicle_rear"}, )
          TEST: ({"rec_dataset_path_file": "/xxx/dataset.yaml", "task_name": "vehicle_rear"},)
        ```
    Dataset 的信息用字典表示，有两个字段 rec_dataset_path_file 和 task_name。其中，rec_dataset_path_file 表示辅助文件 dataset.yaml 的路径，task_name 表示数据集类别的名字。
    
    辅助文件 dataset.yaml 格式如下:
    ```python
    vehicle_rear:
      - common:  # 数据集的基本信息，适用任务、class_id 信息等
          task_type: "detection"
          class_id: 5
          category: 0
          to_rgb: True
        train:  # 训练集，包含 rec 文件路径和 anno 文件路径
          data_path: "/xxxxx/vehicle_rear/detection/720_v1/rec/train.rec"
          anno_path: "/xxxxx/vehicle_rear/detection/720_v1/rec/train.json"
        test:   # 测试集，包含 rec 文件路径和 anno 文件路径
          data_path: "/xxxxx/vehicle_rear/detection/720_v1/rec/val.rec"
          anno_path: "xxxxxx/vehicle_rear/detection/720_v1/rec/val.json"

      - common:
          task_type: "detection"
          class_id: 5
          category: 0
          to_rgb: True
        train:
          data_path: "/xxxxx/rear/detection/720_v1/rec/train.rec"
          anno_path: "/xxxxx/rear/detection/720_v1/rec/train.json"
        test:
          data_path: "/xxxxx/rear/detection/720_v1/rec/val.rec"
          anno_path: "xxxxxx/rear/detection/720_v1/rec/val.json"
    ```
    
2. 使用 HAT 里的 `tools/detectron2_train_net.py` 启动训练
   
   detectron2 的训练入口默认是 detectron2 里的 `tools/train_net.py`，为了densebox dataset适配需要，HAT 提供了 `tools/detectron2_train_net.py` 可替代 detectron2 原有的 `tools/train_net.py`，使用方式也和detectron2 完全一致。
   
   HAT `tools/detectron2_train_net.py` 和 detectron2 `tools/train_net.py` 的区别就是 `Trainer` 略有不同，其他完全相同：
    ```python
    class Trainer(DefaultTrainer):
    ...
        @classmethod
        def build_train_loader(cls, cfg):
            # 改成调用 hat.data.datasets.densebox2detectron2.build_data_loader
            # hat 里的 build_data_loader只会在使用 denseboxdataset config 配置形式时有效，
            # 其他情况下都会使用 detection2 里原本的 build_{train/test}_dataloader 接口
            # 这里改成调用hat里的 build_data_loader，不会对原本的 detection2 框架有影响
            return build_data_loader(cfg, mode="train")

        @classmethod
        def build_test_loader(cls, cfg, dataset_name):
            # 同上, 改成调用 hat.data.datasets.densebox2detectron2.build_data_loader
            return build_data_loader(cfg, dataset_name, mode="test")
   ```

 
