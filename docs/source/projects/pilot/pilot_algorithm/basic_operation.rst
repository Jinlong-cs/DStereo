.. _hat-pilot-basic-op:

基础操作介绍
==================================
通过前面的大致介绍，相信您对Pilot多任务感知算法框架有了简单的认识。之前的介绍都是准备工作，
让您对算法包的内容有一个全局视角的了解。接下来是算法框架中的基础操作介绍，主要介绍如何让训练框架运行起来、
如何评测完成训练的模型，如何可视化模型推理结果。

值得提醒的是，本节内容较为重要，请您仔细阅读。

.. _train-operations-of-multitask:

.. include:: ../../../../../projects/pilot/tools/train/README.md
  :parser: myst_parser.docutils_

.. _eval-operations-of-multitask:

.. include:: ../../../../../projects/pilot/tools/eval/README.md
  :parser: myst_parser.docutils_


.. _hat-pilot-model-compile:

.. include:: ../../../../../projects/pilot/dev/publish/README.md
  :parser: myst_parser.docutils_
  :end-before: # 模型推理结果输出


.. _infer-operations-of-multitask:

模型推理结果可视化
----------------------------

在此介绍推理结果可视化的操作，此处仅支持本地运行。

在hat项目根目录, 使用 ``python3 tools/predict.py``, Example:

.. code-block:: shell

    config_path=projects/pilot/configs/{model_type}/vis_multitask.py
    eval_stage=sparse_3d_freeze_bn_2


    python3 tools/predict.py \
            --config ${config_path} \
            --stage ${eval_stage} \
            --device-ids 0 \

对应参数有:

- `--config`: 评测config本地路径, 相对于hat根目录或绝对路径
- `--stage`: 需要评测的模型stage, 支持训练过程中的各个阶段模型评测
- `--device-ids`: 本地预测使用gpus
