# CARP (Cocktail AlgoRighm Package) 

The Cocktail Algorithms that solve Cocktail Party Problem!

CARP包是服务于视觉加语音的多模语音算法训练库。


## 环境配置

### 1. 配置HAT环境

根据 [环境准备](http://model.aidi.hobot.cc/api/docs/3025/HAT/1.1.0.dev202205301042-0a96ce0/html/quick_start/installation.html#id2) 
或者 [installation instructions](http://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/blob/master/docs/source/quick_start/installation.md)
进行HAT的环境配置; 在 dev051 上比较快捷的方式是(**记得用虚拟环境**)：

``` bash 
vitrualenv py36hat --python=python3.6
cd ${HAT_ROOT}
make run-env-cu111
```

### 2. 配制CARP环境

除了 HAT 的环境之外，CARP 有部分代码不适合放在根目录下，例如

1. 没有充分优化和沉淀的代码
2. 只有 CARP 目录下会使用的环境
3. 部分工具使用中才依赖的库，不使用不依赖的库

这类代码或者环境放入根目录可能会影响整体的CICD。
处于风险可控的角度考虑，在CARP下进行管理。
配置环境的方式是：


``` bash
cd ${CARP_ROOT}
make carp-extra-env
```

### 3. 快速设置PYTHONPATH

对于 HAT、CARP 和 WeNet 等的代码库的使用，通过配置 ``PYTHONPATH`` 的方式使用。
在 CARP 目录下提供了 ``.bashrc`` 文件，快速配置 ``PYTHONPATH``。使用方式：

``` bash 
source ${CARP_ROOT}/.bashrc
```

### 4. 新机器需要绑定 bucket (备选)

考虑到部分数据可能需要使用 bucket 上的数据，因此在新机器上需要手动配置 bucket 环境。

``` bash 
make carp-bucket
```

## TODO


## WENET INSIDE

