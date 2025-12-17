Pilot周视算法
==========================
.. toctree::
   :maxdepth: 1
   :numbered:

   pilot_algorithm/basic_operation.rst
   pilot_algorithm/develop_pipeline.rst
   pilot_algorithm/perception_algo_intro.rst
   pilot_algorithm/model_structure.rst
   pilot_algorithm/perception_algo_pipeline.rst
   pilot_algorithm/annotation_transformer.rst
   pilot_algorithm/train_framework.rst
   pilot_algorithm/train_configs.rst
   pilot_algorithm/tutorial.rst

其中:

* 方案层面：:ref:`pilot-algo-method` 介绍了模型涉及的各种任务，而 :ref:`pilot-model-structure` 则介绍了各种模型的结构，之后在 :ref:`pilot-algo-pipeline` 中则从感知系统的角度将多个模型串联起来，介绍模型整体的使用方式。

* 框架层面：:ref:`training-framework` 介绍了多任务模型的训练架构。

* 实现层面：:ref:`pilot-config-intro` 介绍了整个算法方案的config实现， :ref:`pilot-config-develop` 则一步步展示了如何在现有实现的基础上，如何给多任务模型加一个新任务。

* 使用层面：:ref:`hat-pilot-basic-op` 介绍了各种操作方式。
