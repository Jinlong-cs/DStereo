# 如何使用 Makefile

为了方便用户本地开发和调试，`HAT` 提供了一份[Makefile](http://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/blob/master/Makefile) 文件，可以让用户快速的执行一些命令，结合文件中的具体内容，下面对这些命令的使用方法和功能做一个详细的介绍。  


## 命令介绍

`USER_BASE`: 用于设置 `pip3 install` 需不需要加 `--user`，默认不加，如果需要的话，只需设置 `USER_BASE=--user`。     

`make basic-data`: 运行 `make basic-data` 命令，用于准备数据集和部分预训练模型，这是保证所有示例模型可以成功运行的第一步。  

`make clean-basic-data`: 运行 `make clean-basic-data` 命令，用于删除 `make basic-data` 生成的数据集和预训练模型的软链接。

`make env`: 运行 `make env` 命令，用于安装开发需要的环境依赖。  

`make lint`: 运行 `make lint` 命令，用于检查当前目录下的代码是否符合已有的规范。  

`make isort`: 运行 `make isort` 命令，用于检查代码的 `import` 顺序是否满足规范。

`make black`: 运行 `make black` 命令，用于检查代码是否满足 `black` 规范。

`make flake8`: 运行 `make flake8` 命令，用于检查代码是否满足 `flake8` 规范。

`make pydocstyle`: 运行 `make pydocstyle` 命令，用于检查 `docstring` 文档是否满足规范。

`make doc`: 运行 `make doc` 命令，用于编译和生成文档。  

`make clean-doc`: 运行 `make clean-doc` 命令，清空生成的文档。  

`make run-env-cu116`: 运行 `make run-env-cu116` 命令，安装所有基于 `cuda11.6` 的环境依赖，包括 `torch`、`torchvision`、`horizon_plugin_pytorch` 等等。 

`make collect-env`: 运行 `make collect-env` 命令，查看所有的环境依赖。通常在因为环境问题导致代码无法运行的时候可以有助于快速的定位问题。  

`make unit-test`: 运行 `make unit-test` 命令，跑所有的单元测试用例。通常都需要先 `make basic-data` 成功准备数据集之后，才可以保证 `make unit-test` 可以正常跑通。  

`make intergration-tests`: 运行 `make intergration-tests` 命令，跑所有的集成测试用例。同样需要先运行 `make basic-data` 准备数据集之后，才可以成功运行。

`make wheel`: 运行 `make wheel` 命令，用于生成 `HAT` 的 `wheel` 包。  

`make clean-wheel`: 运行 `make clean-wheel` 命令，用于删除 `make wheel` 生成的 `wheel` 包。

