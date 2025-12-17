
# 在 mmdetection 中使用 HAT DenseboxDataset

HAT 提供了兼容接口，可在社区框架 mmdetection 中通过config方式直接调用 HAT 中的 `DenseboxDataset`。
只需在 mmdetection 的 config 中做如下配置即可:

```python
# 1. 要使用 custom_imports，导入
custom_imports = dict(
    imports=["hat.data.datasets.densebox2mmdet"]
)

# 2. 数据集设置CLASSES, 需要是 list
classes = ['vehicle_rear']
data = dict(

    train=dict(
        # 3.1 type 设置成 DenseboxDataset2MMDet
        type="DenseboxDataset2MMDet",
        # 3.2 数据集设置 HAT_DATASET
        # 设置字段和 HAT `DenseboxDataset` config 配置保持一致
        dataset=dict(
                type="DenseboxDataset",
                data_path="./vehicle_rear/detection/720_v1/rec/train.rec",
                anno_path="./vehicle_rear/detection/720_v1/rec/train.json",
                task_type="detection",
                class_id=5,
                category=0,
                to_rgb=True,
         ),
        classes=classes,
        # 3.4 使用 MMdetection 的 Transform
        pipeline=[
                # 注意: 不要使用 `LoadImageFromFile`
                # dict(type="LoadImageFromFile"), 
                dict(type="LoadAnnotations", with_bbox=True),
                ...
            ],
        
        ),

    val=dict(
        # 同理，同上，改成 val 的数据集信息即可
        type="DenseboxDataset2MMDet",
        dataset=dict(
                    type="DenseboxDataset",
                    data_path="./vehicle_rear/detection/720_v1/rec/val.rec",
                    anno_path="./vehicle_rear/detection/720_v1/rec/val.json",
                    task_type="detection",
                    class_id=5,
                    category=0,
                    to_rgb=True,
            ),
            classes=classes,
            pipeline=[
                    # 注意: 不要使用 `LoadImageFromFile`
                    # dict(type="LoadImageFromFile"), 
                    dict(type="LoadAnnotations", with_bbox=True),
                    ...
                ],
        
        ),
        
    ),
```

如果是使用多个数据集(多个 `rec/annotations`)，把 `dataset` 字段改成 `list` 即可, 例如:

```python
...
# Dataset A
dataset_A = dict(
    type="DenseboxDataset",
    data_path=dataset_dir["train_rec_file"],
    anno_path=dataset_dir["train_json_file"],
    task_type="detection",
    class_id=5,
    category=0,
    # note: Normalize needs to be after ToTensor if `to_yuv`==True
    to_rgb=True,
)

# Dataset B
dataset_B = dict(
    type="DenseboxDataset",
    data_path=dataset_dir["train_rec_file_full"],
    anno_path=dataset_dir["train_json_file_full"],
    task_type="detection",
    class_id=5,
    category=0,
    # note: Normalize needs to be after ToTensor if `to_yuv`==True
    to_rgb=True,
)

data = dict(
    train=dict(
            type="DenseboxDataset2MMDet",
            classes=classes,
            dataset=[dataset_A, dataset_B],   # 修改这里，改成 list
            pipeline=[
               ...
            ],
        ),
    val=dict(
        ...
    )
)
...
```
