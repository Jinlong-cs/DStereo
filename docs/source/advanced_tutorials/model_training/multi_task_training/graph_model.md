(hat-graph-model)=
# HAT中的多任务模型架构

## Why

在多任务模型的相关应用中，我们发现以下几个较为重要的应用需求：

1. 支持灵活声明计算图的拓扑结构。尤其是需要支持各任务的计算结构分开声明、组合使用，以提高各任务子结构的复用性，提高整体的灵活性。
2. 能够方便获取全量任务任意子集对应的子计算图，方便基于子图做forward-backward操作。这里有几种情况，
   - 需要使用全量任务的部分子任务集合做推理，因此需要获取相应任务对应的计算图。
   - 数据中可能并不包含所有任务的标注，需要基于部分任务对应的计算结构做训练。
   - 模型结构本身较为复杂，所有任务都展开的情况下会占用大量显存。需要通过分步基于各子图做forward-backward的方式来进行更高效地训练。

## 初步实现{py:class}`GraphModel <hat.models.structures.GraphModel>`

为满足上述需求，我们基于`torch`原生的`nn.Module`和内部开发的`HATBC workflow`，实现了{py:class}`GraphModel <hat.models.structures.GraphModel>`这个模型类。

其初始化参数主要有
1. `nodes`: 声明计算图内的计算节点（可以是`nn.Module`或一个python函数），每个node有一个全局唯一的名称。
2. `inputs`: 描述模型的输入，用来帮助后续通过trace来构建模型对应的计算图。
3. `topology_builder`: 用来描述模型计算图的拓扑结构的callback。其输入为上面的`inputs`和 `nodes`，输出则是以`Mapping`形式封装的模型输出，保证各输出都能通过固定的`key`来获取。

初始化阶段，首先基于`HATBC workflow`提供的trace功能，将`topology_builder`表示的计算逻辑转化为计算图。通过维护`topology_builder`的输出中`key`和`value`的对应关系，即可得到各计算子图。随后再将 `nodes`中涉及的 ``nn.Module`` 实例注册至{py:class}`GraphModel <hat.models.structures.GraphModel>`实例下，以支持torch原生的基于`nn.Module`的模型参数管理。

运行时，随着正常输入模型的数据，还可以加入一个`out_names`变量，用以获取对应子图输出的结算结果。

## 不足

{py:class}`GraphModel <hat.models.structures.GraphModel>`一定程度上满足了上述的两个需求，但也有些不足，并引入了新问题：

1. 多次存在的相同计算流程无法自动合并，如果重复声明，会导致计算量增大、显存占用变多。这也导致，在原本的设定下，一个共享部分结构的多任务模型，其结构可能会分布在不同的config文件中，被分别声明。

2. 将`nn.Module`统一视为`node`，计算图内无法无法像目前比较流行的使用方式一样，使用较为通用的抽象structure结构（典型的如{py:class}`TwoStageDetector<hat.models.structures.detectors.TwoStageDetector>`或{py:class}`Segmentor<hat.models.structures.Segmentor>`等)，使得代码层面不太方便通过。此外，这还导致了使用{py:class}`GraphModel <hat.models.structures.GraphModel>`描述的多任务模型在编程风格无法与常规的pytorch包对齐，与单任务模型的构建方式完全不同。

上述的两个问题，也可以通过下面的例子形象体现。

```python
# task1.py
task_name = "task1"
inputs = {}  # something

head=dict(
   type=...,
   ...
)

def topo_builder(nodes, _inputs, feats):

   # filter by inputs keys
   inner_inputs = {k: _inputs[k] for k in inputs}
   name2out = OrderedDict()
   out_module = nodes[f"{task_name}_head"]
   name2out.update({task_name: out_module(feats, inner_inputs)})
   return name2out
#


# task2.py
task_name = "task2"
inputs = {}  # something

head=dict(
   type=...,
   ...
)

def topo_builder(nodes, _inputs, feats):

   # filter by inputs keys
   inner_inputs = {k: _inputs[k] for k in inputs}
   name2out = OrderedDict()
   out_module = nodes[f"{task_name}_head"]
   name2out.update({task_name: out_module(feats, inner_inputs)})
   return name2out
#


# multitask.py
from task1 import topo_builder as task1_topo
from task1 import head as task1_head
from task2 import topo_builder as task2_topo
from task2 import head as task2_head

backbone = dict(
   type="...",
   ...,
)  # some backbone module
neck = dict(
   type="...",
   ...,
) # some backbone module


def topo_builder(nodes, inputs):
    name2out = OrderedDict()

   backbone_feats = nodes["backbone"](inputs["img"])
   neck_feats = nodes["neck"](backbone_feats)

   for builder in [task1_topo, task2_topo]:
      name2out.update(
            builder(
               nodes=nodes,
               _inputs=inputs,
               feats=neck_feats,
            )
      )
   return name2out
#

model = dict(
   type="GraphModel",
   nodes=dict(
      backbone=backbone,
      neck=neck,
      task1_head=taks1_head,
      task2_head=task2_head,
   )
   inputs=...,
   topology_builder=topo_builder,
)
```

## 改进实现{py:class}`MultitaskGraphModel <hat.models.structures.MultitaskGraphModel>`

为了克服上面列举的不足，我们在{py:class}`GraphModel <hat.models.structures.GraphModel>`的基础上，开发了改进版的{py:class}`MultitaskGraphModel <hat.models.structures.MultitaskGraphModel>`。其主要的亮点在于：

1. 弃用`topology_builder`，换成用`nn.Module`表示模型结构，这种方式显著提高了框架层面的复用率，并让多任务和单任务模型在声明方式上更为一致。通过是否在初始化参数中添加`node_name`的方式（参考[ModulePatch文档](./module_patch.md)的相应内容），来显式声明对应`Module`是否需要成为计算图中的节点，以方便构建计算图。
2. 加入自动的相同计算流合并。将所有可合并（op实例相同、args相同、输入相同）的节点进行合并，使得重复声明的计算流能够共享使用。模型的声明更为容易，多任务模型可以在各任务的视角下分别定义完整模型结构。

与上面例子相同的模型结构，在新的方式下，可以声明为：
```python
# somewhere in HAT
@OBJECT_REGISTRY.register
def NaiveStructure(nn.Module):
   def __init__(self, backbone, neck, head):
      self.backbone = backbone
      self.neck = neck
      self.head = head
   
   def forward(self, data, labels):
      backbone_feats = self.backbone(data["img"])
      neck_feats = self.neck(backbone_feats)
      return self.head(neck_feats, labels)
#


# common.py
backbone = dict(
   type="...",
   ...,
   nodel_name="backbone",
)  # some backbone module
neck = dict(
   type="...",
   ...,
   node_name="neck",
) # some backbone module

# task1.py
from common import backbone, neck

task_name = "task1"
inputs = {}  # something

model = dict(
   type="NaiveStructure",
   backbone=backbone,
   neck=neck,
   head=dict(
      type=...,
      ...
      node_name=f"{task_name}_head",
   ),
)
#


# task2.py
from common import backbone, neck

task_name = "task2"
inputs = {}  # something

model = dict(
   type="NaiveStructure",
   backbone=backbone,
   neck=neck,
   head=dict(
      type=...,
      ...
      node_name=f"{task_name}_head",
   ),
)
#


# multitask.py
from task1 import model as task1_model
from task2 import model as task2_model

model = dict(
   type="MultitaskGraphModel",
   inputs=...,
   task_inputs=...,
   task_modules=dict(
      task1=task1_model,
      task2=task2_model,
   ),
)
#
```

可以看出，在新版的实现中，多任务模型结构的声明整体上做到了与单任务的对齐，更符合pytorch使用者的编程习惯。

## 方案的局限性

回到最初的动机，可以看到，上述的两种方法，都基本从根本上改善了问题。但由于引入了新的静态图引擎，也导致整体的局限性，在此也列举一下，请开发者、使用者注意：

1. 新引入的`HATBC workflow`静态图通过trace方式搭建，因此也引入了占位符的概念。需要为每个输入预先指定对应变量。
2. 由于静态计算图的引入，无法像往常一样在torch中任意位置通过打断点的方式debug了（在GraphModel中无明显差异）。目前，只支持node内部（对于MultitaskGraphModel，只有声明config中参数有`node_name`的module才算计算图node）的断点调试。