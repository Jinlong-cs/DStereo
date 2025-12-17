# 如何在HAT中开发metric

## `HAT` metric整体介绍

在机器学习的语境中，metric的作用是用量化的指标来描述模型的效果，其输入一般是一系列成对的模型预测结果和真值，在一些情况下，也可能只有模型预测值。

`HAT`中的metric全面基于[torchmetrics](https://github.com/PyTorchLightning/metrics/)搭建。其主要优势为对分布式计算的原生支持，能够充分利用gpu资源，实现高效的计算。

在`HAT`中，一般情况下metric会在`MetricUpdater` ([hat/callbacks/metric_updater.py](http://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/blob/master/hat/callbacks/metric_updater.py#L203)) 这个callback中被调用。具体来讲，对于metric，在每一个batch的数据运行完后，执行`update`操作更新内部统计量，而在所有数据跑完后，再使用`get`方法根据metric的内部统计量得到最后的结果。

## 基类介绍

`HAT`中metric的基类是`EvalMetric`([hat/metrics/metric.py](http://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/blob/master/hat/metrics/metric.py#L21))，基于`torchmetrics`中的基类[`Metric`](https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/metric.py#L45)实现。`EvalMetric`提供了许多可继承的标准化接口，帮助开发者快速搭建自己的支持分布式运行的metric。

`EvalMetric`也是`torch.nn.Module`的子类，因此也继承了相应的外部调用的接口（如`forward`和`to`)和内部状态的维护方式(使用`buffer`）。

基类初始化和各接口的相关说明请参考API reference中的[相关内容](../../api_reference/metrics.html#hat.metrics.EvalMetric)，在此就不再重复介绍。接下来，会用一个例子来向开发者讲述如何基于`EvalMetric`快速实现自己的metric。

## 快速实现一个自定义的metric

这一节我们介绍如何在`HAT`中开发`Mean IoU`这个metric。如果阅读者对这一部分提到的一些细节感到难以理解，请尝试先阅读我们提供的[API文档](../../api_reference/metrics.html#hat.metrics.EvalMetric)。如果依然有问题，还可以到相应的问答群组去提问。

### 初始化metric类

```python
class MeanIOU(EvalMetric):
    """Evaluation segmentation results.

    Args:
        seg_class: a list of classes the segmentation dataset includes，
            the order should be the same as the label.
        name: name of this metric instance for display, also used as
            monitor params for Checkpoint.
        ignore_index: the label index that will be ignored in evaluation.
        global_ignore_index: the label index that will be ignored in
            global evaluation, egs:mIoU,mAcc,aAcc.
        verbose:  Whether to return verbose value for aidi eval, default
            is False.

    """
    def __init__(
        self,
        seg_class: List[str],
        name: str = "MeanIOU",
        ignore_index: int = 255,
        global_ignore_index: int = 255,
        verbose: bool = False,
    ):
        self.num_classes = len(seg_class)
        self.seg_class = seg_class
        self.name = name
        super(MeanIOU, self).__init__(name)
        self.ignore_index = ignore_index
        self.global_ignore_index = global_ignore_index
        self.verbose = verbose
```

如docstring所示，这里的大部分初始化参数都来自于`Mean IoU`计算逻辑，只有`name`是`EvalMetric`这个基类所关心的内容。

### 初始化内部状态

```python
    def _init_states(self):

        self.add_state(
            "intersect",
            default=torch.zeros((self.num_classes,)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "union",
            default=torch.zeros((self.num_classes,)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "pred_label",
            default=torch.zeros((self.num_classes,)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "label",
            default=torch.zeros((self.num_classes,)),
            dist_reduce_fx="sum",
        )
```

这一步做的是初始化metric计算过程所需要的内部状态。需要注意的是，为了保证相应状态能够被跨设备（gpu）同步，请使用`self.add_state`方法。而方法中`dist_reduce_fx`接口，则声明了对应state在分布式同步后的reduce方法。

### update

```python
    def update(
        self,
        label: torch.Tensor,
        preds: Union[Sequence[torch.Tensor], torch.Tensor],
    ):
        """
        Update internal buffer with latest predictions.

        Note that the statistics are not available until
        you call self.get() to return the metrics.

        Args:
            preds: model output.
            label: gt.

        """
        # only one pred and one gt used in MeanIOU calculation.
        pred_label = _as_list(preds)[0].detach()

        mask = label != self.ignore_index
        pred_label = pred_label[mask].float()
        label = label[mask].float()

        intersect = pred_label[pred_label == label]

        area_intersect = torch.histc(
            intersect, bins=self.num_classes, max=self.num_classes - 1
        )
        area_pred_label = torch.histc(
            pred_label, bins=self.num_classes, max=self.num_classes - 1
        )
        area_label = torch.histc(
            label, bins=self.num_classes, max=self.num_classes - 1
        )
        area_union = area_pred_label + area_label - area_intersect

        self.intersect += area_intersect
        self.union += area_union
        self.pred_label += area_pred_label
        self.label += area_label
```

和上面初始化状态阶段的代码对比，我们可以看出来，通过调用`self.add_state`做初始化后，可以使用state的名字（如`self.intersect`、`self.union`等）来访问相应内容。

这里具体的计算逻辑只是一些简单的直方图统计，就不展开讲了。由于计算本身都是基于`torch.Tensor`实现的，所以非常容易理解。`update`方法的特点就是，只更新内部状态，而没有返回值。

需要强调的是，在分布式情况下，`update`方法内一般无法拿到别的设备（gpu）上的数据，因此我们计算的都是local state。

### compute

```python
    def compute(self):
        """Get evaluation metrics."""

        def _del_tensor_ele(x: torch.Tensor, index: int):
            """Delete the element at the specified index."""

            slice1 = x[:index]
            slice2 = x[index + 1 :]
            return torch.cat((slice1, slice2), dim=0)

        all_acc = (
            _del_tensor_ele(self.intersect, self.global_ignore_index).sum()
            / _del_tensor_ele(self.label, self.global_ignore_index).sum()
        )
        acc = self.intersect / self.label
        iou = self.intersect / self.union

        summary_str = "~~~~ %s Summary metrics ~~~~\n" % (self.name)
        summary_str += "Summary:\n"
        line_format = "{:<15} {:>10} {:>10} {:>10}\n"
        summary_str += line_format.format("Scope", "mIoU", "mAcc", "aAcc")

        miou = _del_tensor_ele(iou, self.global_ignore_index).mean()
        macc = _del_tensor_ele(acc, self.global_ignore_index).mean()
        iou_str = "{:.2f}".format(miou.cpu().item() * 100)
        acc_str = "{:.2f}".format(macc.cpu().item() * 100)
        all_acc_str = "{:.2f}".format(all_acc * 100)
        summary_str += line_format.format(
            "global", iou_str, acc_str, all_acc_str
        )

        summary_str += "Per Class Results:\n"
        line_format = "{:<15} {:>10} {:>10}\n"
        summary_str += line_format.format("Class", "IoU", "Acc")

        for i in range(self.num_classes):
            iou_str = "{:.2f}".format(iou[i].cpu().item() * 100)
            acc_str = "{:.2f}".format(acc[i].cpu().item() * 100)
            summary_str += line_format.format(
                self.seg_class[i], iou_str, acc_str
            )
        logger.info(summary_str)

        if self.verbose:
            return miou, macc, all_acc, iou, acc, self.seg_class
        else:
            return miou
```

`compute`中，一般会基于之前更新好的内部状态得到最终的结果。这里的实现中，还顺便把一些重要信息以log的方式展示了出来。

这里的重点是，在`compute`中，metric的内部状态已经做了跨设备的同步，因此都是全局统计量。

## 如何配置使用已定义的metric

一般来讲是放在`MetricUpdater`的`metrics`参数中使用，一个例子是

```python
dict(
    type="MetricUpdater",
    metrics=[
        dict(
            type="MeanIOU",
            seg_class=seg_class,
            name="MeanIOU",
            ignore_index=ignore_index,
        ),
    ],
    filter_condition=lambda x: x[1] == task_name,
    metric_update_func=get_update_metric(is_train=False),
    log_prefix="Validation " + task_name,
    step_log_freq=-1,
)
```

其中，`MetricUpdater`的具体使用可以参考[API文档](../../api_reference/callbacks.html#hat.callbacks.MetricUpdater)的内容。

此外，当在`Validation`中使用metrics时，在分布式的情况下，请注意将相应`Dataloader`的`sampler`项设置为`DistributedSampler`，比如

```python
val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ImageNet",
        data_path="./tmp_data/imagenet/val_lmdb/",
        transforms=[
            dict(type="TorchVisionAdapter", interface="Resize", size=256),
            dict(type="TorchVisionAdapter", interface="CenterCrop", size=224),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=5,
)
```

以避免数据在不同device上被重复读取、使用。

## FAQ （持续更新）

- 问：metric中的输入数据是跑在什么上面的？gpu还是cpu？metric里定义的各种state和输入数据会在一样的设备上吗？

  1. 每个metric具体的输入内容由使用者决定，详情可以参考`MetricUpdater`的[相关文档](../../api_reference/callbacks.html#hat.callbacks.MetricUpdater)。因此输入在哪个设备上取决于其来源，无法一概而论。但一般情况下，metric的输入是模型输出和数据标签，应该会在训练设备（如gpu上）。
  2. 用`self.add_state`方式添加的状态在实现上就是`torch.nn.Module`的buffer，因此在初始化阶段，就会作为metric的一部分被放入相应设备（执行`metric.to(device)`)。而以其他方式所添加的状态，则需要用户自己确定。

- 问：在`__init__`中和`_init_states`中所初始化的内部状态有什么区别？

  1. 重点不在于内部状态在哪里定义，而在于定义的形式。通过`self.xxx = yyy`直接定义的参数可以是任意的数据格式，但无法被自动跨设备同步，因此一般存放静态或者设备无关的变量。而通过`self.add_state`方法做初始化的内部状态，其数据格式必须是`torch.Tensor`，会自动在需要时被跨设备同步。
  2. 如果有通过`self.xxx = yyy`所定义的参数在`update`中会被更新，则需要扩展`reset`方法去做相应的重置，否则`self.reset`只会重置通过`self.add_state`初始化的状态。

- 问：如果我暂时没想好自己的metric应该怎么以分布式的方式实现，应该怎么做？`HAT`中的`metric`模块支持非分布式的实现方式吗？

    这里可以参考API reference中的[相关内容](../../api_reference/metrics.html#hat.metrics.EvalMetric)，首先不使用`self.add_state`方法定义内部状态，而是直接在`__init__`中直接实现。其次是手动重载`get`方法，跳过`compute`步骤，直接计算结果。
