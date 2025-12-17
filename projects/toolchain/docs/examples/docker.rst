Docker 镜像说明
==================


1. 如何启动
------------

为方便使用 `HAT` 算法包，我们提供了一键启动 `docker` 环境的命令，以简化使用过程中的环境搭建步骤。用户在获取发布包后先进入发布包所在的路径，然后根据 `《地平线J5 AI芯片工具链用户手册》` 的 `使用Docker环境`
一节的内容完成基础Docker环境安装、添加Docker组用户、拉取Docker镜像之后，只需使用以下命令，就可以获取成功运行 `HAT` 示例的所有环境。

  .. code-block:: bash

      nvidia-docker run -it --shm-size="15g" -v `pwd`:/open_explorer openexplorer/ai_toolchain_centos_7_j5:{version} /bin/bash

  .. note::

      此处的版本号{version}仅为示例，请将其替换为您实际获取到的镜像版本号。

其中 `openexplorer/ai_toolchain_centos_7_j5:{version}` 是镜像名，`-v` 用于将本地的路径挂载到 `docker` 路径下。


2. 获取执行脚本
----------------

成功启动 `docker` 环境之后，就可以进入执行脚本的目录获取执行脚本，即：

  .. code-block:: bash

      cd /open_explorer/ddk/samples/hdlt/scripts/

你会看到 `configs` 和 `tools` 两个文件夹目录。所有环境都拥有之后，你就可以按照后面的训练教程来一步步的使用 `HAT` 训练出一个定点模型。
