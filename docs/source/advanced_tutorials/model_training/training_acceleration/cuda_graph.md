# 如何使用 CUDA-Graph

CUDA-graph 是NVIDIA官方提供的一种融合kernel launch次数的工具。

常用的GPU程序，需要多次调用cpu去逐个launch gpu kernel，以完成整个kernel序列的启动。整个过程中，CPU启动开销也占用了很大的一部分。而CUDA-Graph是将整个计算流程定义为一个图而不是单个操作的列表。 最后通过提供一种由单个CPU操作来启动图上的多个GPU操作的方式减少kernel的启动开销，进而解决CPU启kernel时开销大的问题。

其中，CUDA-Graph的NVIDIA官方示例可以参考[NVIDIA文档](https://developer.nvidia.com/blog/cuda-graphs/)。torch社区集成CUDA-Graph的方法可以参考[torch文档](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/).

HAT中提供了一些基础示例，可以帮助用户快速开发和使用CUDA-Graph。以分类模型为例，分别提供partial graph和whole graph两个方法的设置。

## partial graph

partial graph主要是借助`torch.cuda.make_graphed_callables`的方法，对模型中的部分结构转cuda graph。

HAT中的设置方法：
1. 环境变量的设置，这一步主要是保证CUDA-Graph的相关设置可以全局生效，如NCCL的环境变量，DDP的stream启动等等。
``` bash
 export HAT_USE_CUDAGRAPH=1
```
2. 模型中partial graph的设置。分类模型已经默认集成了设置方法，直接在Classifier中设置cuda_graph=True即可，其他的模型可以根据类似的方法设置即可。具体实现的方法如下：
``` python
if self.graph_backbone is None:
    self.graph_backbone = torch.cuda.make_graphed_callables(
        self.backbone, (image,), num_warmup_iters=11
    )
    preds = self.graph_backbone(image)
else:
    preds = self.graph_backbone(image)
```

## whole graph

whole graph就是对模型整个forward/backward，可能还包括optimizer更新的全流程转换成cuda graph。这种方法的成本也是比partial graph更高的。但这种方法同时可以capture NCCL的相关操作，理论上是可以比partial graph更加高效的。

HAT中的设置方法：
1. 环境变量的设置，和partial graph是一样的功能。
``` bash
 export HAT_USE_CUDAGRAPH=1
```
2. 使用默认集成capture whole graph的功能的`CudaGraphBatchProcessor`。绝大部分参数和`BasicBatchProcessor`是一样的，少量不支持的功能已经从接口里删除。
``` python
batch_processor = dict(
    type="CudaGraphBatchProcessor"
    ...
)
```
由于whole graph的通用性不高，所以这里只提供了`CudaGraphBatchProcessor`这一种cuda-graph的方法。实际过程需要根据训练的模式和CUDA-Graph的限制条件，做更精细的调整。
