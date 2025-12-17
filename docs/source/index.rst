Horizon Algorithm Toolkit (海图)
================================

Horizon Algorithm Toolkit是地平线提供基于Pytorch的深度学习训练
工具，由Pytorch plugin和HAT算法包两部分组成。

Pytorch plugin是基于Pytorch开发的一套量化算法工具，其量化算法与地平线芯片深度耦合，利用该工具训练得到的量化模型均可以正常编译和运行
在地平线BPU上。

HAT算法包是基于Pytorch和Pytorch plugin的接口开发的一套高效且用户友好的AI算法工具。同时它还可以提供包含分类，检测等常见的图像
任务的SOTA(state-of-the-art)深度学习模型。


.. toctree::
   :maxdepth: 1
   :caption: Introduction

   introduction/introduction.md

.. toctree::
   :maxdepth: 1
   :caption: Quick Start

   quick_start/installation.md
   quick_start/launcher.md
   quick_start/eval.md
   quick_start/docker.md

.. toctree::
   :maxdepth: 1
   :caption: Notes

   notes/terminology.md
   notes/hat_env.md

.. toctree::
   :maxdepth: 1
   :caption: Basic Concepts

   basic_concepts/framework.md
   basic_concepts/registry.md

.. toctree::
   :maxdepth: 1
   :caption: Basic Tutorials

   basic_tutorials/config.md
   basic_tutorials/model_convert_pipeline.md
   basic_tutorials/train_predict.md
   basic_tutorials/load_model.md

.. toctree::
   :maxdepth: 2
   :caption: Advanced Tutorials

   advanced_tutorials/index.rst

.. toctree::
   :maxdepth: 2
   :caption: Useful Tools

   useful_tools/index.rst

.. toctree::
   :maxdepth: 1
   :caption: Framework Compatible

   framework_compatible/index.rst

.. toctree::
   :maxdepth: 1
   :caption: Projects

   projects/index.rst

.. toctree::
   :maxdepth: 1
   :caption: ModelZoo

   model_zoo/model_zoo.md

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api_reference/data.rst
   api_reference/callbacks.rst
   api_reference/engine.rst
   api_reference/models.rst
   api_reference/metrics.rst
   api_reference/profiler.rst
   api_reference/visualize.rst
   api_reference/optimizers.rst

.. toctree::
   :maxdepth: 1
   :caption: FAQs

   faqs/faqs.md

.. toctree::
   :maxdepth: 1
   :caption: ChangeLog

   changelogs/CHANGELOG.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
