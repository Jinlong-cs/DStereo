# 安装

*此文档仅提供基于pytorch2.0.1的环境安装，其他torch版本的环境安装可以结合此文档和 [Makefile](http://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/blob/master/Makefile) 文件*

## 准备环境

1. 需要 python3.8(或python3.10) 和 CUDA 环境（CUDA 版本支持 cuda-11.8）
   这里提供了两种方式准备python环境：

   **方式一**: 通过 `virtualenv` 搭建python虚拟环境：
   * 安装virtualenv：
      ```bash
      pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple virtualenv
      ```
   * 新建环境：
      ```bash
      virtualenv --no-site-packages --python=python3 $name # 新建的环境python版本和系统的python3版本一致
      ```


   * 使用方式：
      ```bash
      # 激活环境
      source $name/bin/activate

      # 退出环境
      deactivate
      ```
   **方式二**: 使用 `conda`:

   * 下载推荐版本的conda:
   
      官方链接下载：
      ```bash
      wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
      ```

      HDFS下载：
      ```bash
      hdfs dfs -get hdfs://hobot-bigdata/user/sitong.chen/conda_install/Miniconda3-4.5.4-Linux-x86_64.sh
      ```
      
      **注:** 建议使用推荐的版本，其他版本的conda中预装的库可能和HAT中的依赖库有冲突，可能会导致HAT安装失败。

   * 安装conda：
      ```bash
      sh ./Miniconda3-4.5.4-Linux-x86_64.sh -b -p conda_env/  # 安装到conda_env文件夹中
      ```

   * 使用conda 新建环境并激活

      ```bash
      # 激活conda base环境
      source conda_env/bin/activate

      # 新建环境
      conda create --yes -n $name python=3.8

      # 进入新建的环境
      conda activate $name

      # 退出环境
      conda deactivate
      ```

2. 需要将 HAT 仓库克隆到本地:
   ```bash
   # step1: clone HAT 
   git clone git@gitlab.hobot.cc:ptd/algorithm/ai-platform-algorithm/HAT.git
   
   # step2: 进入 HAT 根目录, 检查所在分支
   cd HAT
   git branch  # 默认在 master 分支，建议使用 master
   ```
  
3. 安装所有的依赖库  

    在安装依赖库之前，请先通过终端输入以下命令检查当前 python 虚拟环境和 python 版本是否正确:
    ```bash
    which python3 && python3 --version
    ```
  
   这里提供了两种安装的方式： 
   
    **方式一**: 通过 `make` 命令一键安装  
   `HAT` 提供了一份 [Makefile](http://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/blob/master/Makefile) 文件，用户可以通过简单的 `make` 指令实现快速的环境安装。  
   
   * 基于cuda-11.8的环境，使用以下命令安装：
      ```bash
      make run-env-cu118
      ```
      **注:** 同理，如果你的环境要求使用 `pip install --user` 才能安装依赖，只需使用以下命令安装即可：
      ```bash
      make run-env-cu118 USER_BASE=--user
      ```
      
   **方式二**: 通过 `pip` 命令依次安装  
   a. 安装 Pytorch 2.0.1 、torchvision 0.15.2 和torchaudio 2.0.2，安装后可在终端通过以下命令检测 Pytorch、torchvision 和 torchaudio 版本：
   ```bash
   python3 -c "import torch; print(torch.__version__)"
   ```
   若版本不符合要求，请参照 [此处](https://horizonrobotics.feishu.cn/wiki/wikcnUtqpbxQ1qIx5WNu1mpd1Gg) 安装 Pytorch 2.0.1 、torchvision 0.15.2 和 torchaudio 2.0.2。  
   b. 安装 `horizon_plugin_pytorch`：

   * 基于cuda-11.8的环境，使用以下命令安装：
       ```bash
       pip3 install -U horizon-plugin-pytorch -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/cu118/torch201 -i http://pypi.hobot.cc/simple --extra-index-url http://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc --trusted-host art-internal.hobot.cc
       pip3 install -U horizon-plugin-profiler -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
       ```
   详细的安装命令可以参照 [horizon_pytorch_plugin安装文档](https://horizonrobotics.feishu.cn/wiki/wikcnFzX89YWXJK7i6K81JDa9hh#w9Gk0Q) 。   
   c. 通过以下命令，安装其他依赖库： 
   ```bash
   pip3 install --use-deprecated=legacy-resolver -r requirements.txt -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
   ```
   **注:** horizon-plugin-pytorch在torch1.10.2的环境中可以支持的最高版本是1.10.6.
   


## 安装 HAT

HAT 有三种安装方式:

**方式一**: 设置环境路径

HAT 是纯 Python 算法包，在 `PYTHONPATH` 中添加 `HAT` 根目录路径即可使用:
   ```bash
   # step1: clone HAT 
   git clone git@gitlab.hobot.cc:ptd/algorithm/ai-platform-algorithm/HAT.git
   
   # step2: 进入 HAT 根目录, 检查所在分支
   cd HAT
   git branch  # 默认在 master 分支，建议使用 master
   
   # step3: 可以在 HAT 根目录下的执行 pwd 命令，获取 HAT 的完整路径
   # pwd
   
   # step4: 把获取到的 HAT 完整路径加入到 ~/.bashrc 文件的 PYTHONPATH 中:
   # (假设 HAT_ROOT_DIR_PATH 是通过 pwd 获取的 HAT 目录路径)
   export PYTHONPATH=HAT_ROOT_DIR_PATH:$PYTHONPATH
   
   # step5: 如果是把 HAT 路径添加到 ~/.bashrc 文件中，需执行 source 命令使环境变量生效
   # 如果是在终端中执行的 step4 中的 export 命令的话，请忽略此步
   source ~/.bashrc
   ```
   
   
**方式二**: 源码编译安装

  ```bash
  git clone git@gitlab.hobot.cc:ptd/algorithm/ai-platform-algorithm/HAT.git
      
  # step2: 进入 HAT 根目录, 检查所在分支
  cd HAT
  git branch  # 默认在 master 分支，建议使用 master

  # build and install
  python3 setup.py install
  ```


**方式三**: `pip` 安装

  ```bash
  pip3 install -U horizon-hat -i http://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc      # 最新的release版本
  pip3 install --pre horizon-hat -i http://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc   # 最新的dev版本
  ```


HAT 安装成功后，可通过在终端输入以下命令来验证 HAT 是否安装成功：

  ```bash
  python3 -c "import hat; print(hat.__version__)" 
  ```
 
    
## Docker 运行环境

HAT 以 docker 镜像的形式提供了需要的运行环境，用户在**安装 HAT**后使用 `hat.get_docker_url()` 即可得到镜像的 url。

此方法仅适用于使用以下两种方式安装的 HAT：
- 直接使用 whl 包安装
  - 安装 dev 版本可以得到内部使用的 docker 镜像（镜像中不含 HAT），可用于在 aidi 集群提交训练任务
  - 安装 release 版本可以得到发布使用的 docker 镜像（镜像中包含 HAT）
- 克隆主分支后**立即**在本地编译安装（安装时没有提交任何 commit，安装后才可提交）


## 备注

aidisdk是HAT中的重要依赖，但并不是安装完成之后就可以直接使用，需要配置AIDI相关的环境即可。

如果有AIDISDK相关的环境配置问题，可以参考[AIDISDK文档](http://model.aidi.hobot.cc/api/docs/AIDISDK/latest/html/build/quick_start/index.html)
