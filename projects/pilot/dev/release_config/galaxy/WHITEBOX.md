# Pilot-Galaxy 白盒解压说明

## 解包
运行`openssl des3 -d -k 123456789 -salt -in galaxy_whitebox.tar.gz | tar xzvf - -C unpack_galaxy_whitebox` 将压缩包内容解压到`unpack_galaxy_whitebox`文件夹内

## 加载docker
运行`docker load -i galaxy_whitebox.tar`将docker进行加载

## 进入docker
运行`docker run -it  --gpus all  --shm-size 128g -v {本地pilot_data原始数据集的绝对地址}:/pilot_data_raw docker.hobot.cc/imagesys/hat:pilot-runtime-cu111-hdflow-20230506-torch1102-aidisdk0111-py38-galaxy-whitebox /bin/bash` 进入docker环境


# Pilot-Galaxy 白盒测试脚本使用说明
在上面进入docker后，白盒代码都在/release文件夹内

## 使用方式
由于内嵌aidisdk包，使用前先执行`aidisdk config -t <aidi-token>`进行设置，要不然aidisdk会报错，aidi-token需要在AIDI平台的个人中心里获取
在代码包同级目录，使用`bash whitebox_test.sh`一键运行脚本，即可完成白盒代码的训练+评测+可视化+编译流程测试


## 需要修改的路径
### 数据路径
如果需要修改模型训练评测相关数据，则在对应模型config文件夹下的common.py文件中修改ds_path

Example：如需要修改侧视模型数据，则修改projects/pilot/configs/resize_2_side_bayes/common.py内的ds_path指定的数据读取路径

### 模型路径
测试脚本中，已给出crop/侧视/后视/图像质量模型的checkpoint文件，如果需要在训练或评测时更换模型，则修改HAT_PILOT_MODEL_CHECKPOINT这个环境变量即可


## 训练
在代码包根目录, 使用`python3 tools/train.py`, Example:

        # cd to release_package root
        export HAT_PILOT_MODEL_SETTING=galaxy_x3c_side_lmdb

        config_path=projects/pilot/configs/resize_2_side_bayes/multitask.py
        stage=with_bn


        python3 tools/train.py \
                --config ${config_path} \
                --stage ${stage} \
                --ids 0 \

对应参数有:

- `--config`: 训练config本地路径, 相对于hat根目录或绝对路径
- `--stage`: 指定开始训练的模型stage
- `--ids`: 本地训练使用gpus

此外, 我们通过环境变量确定模型:
- `HAT_PILOT_MODEL_SETTING`: pilot模型的训练数据setting（比如针对galaxy侧视模型，就是galaxy_x3c_side_lmdb)


## 评测
在代码包根目录, 使用`python3 tools/predict.py`, Example:

        # cd to release_package root
        export HAT_PILOT_MODEL_SETTING=galaxy_0233_rear_lmdb
        export HAT_PILOT_EVAL_DATA_SETTING=galaxy_0233_rear_day

        config_path=projects/pilot/configs/resize_2_rear_bayes/val_multitask.py
        stage=sparse_3d_freeze_bn_2


        python3 tools/predict.py \
                --config ${config_path} \
                --stage ${stage} \
                --ids 0 \

对应参数有:

- `--config`: 评测config本地路径, 相对于代码包根目录或绝对路径
- `--stage`: 需要评测的模型stage, 支持训练过程中的各个阶段模型评测
- `--ids`: 本地预测使用gpus

此外, 我们通过环境变量确定模型:
- `HAT_PILOT_MODEL_SETTING`: pilot模型的训练数据setting
- `HAT_PILOT_EVAL_DATA_SETTING`: pilot模型的评测数据setting


## 可视化
在代码包根目录, 使用`python3 tools/predict.py`, Example:

        # cd to release_package root
        export HAT_PILOT_MODEL_SETTING=galaxy_0233_rear_lmdb

        config_path=projects/pilot/configs/resize_2_rear_bayes/vis_multitask.py
        stage=sparse_3d_freeze_bn_2


        python3 tools/predict.py \
                --config ${config_path} \
                --stage ${stage} \
                --ids 0 \

对应参数有:

- `--config`: 可视化config本地路径, 相对于代码包根目录或绝对路径
- `--stage`: 需要可视化的模型stage, 支持训练过程中的各个阶段模型评测
- `--ids`: 本地可视化使用gpus

此外, 我们通过环境变量确定模型:
- `HAT_PILOT_MODEL_SETTING`: pilot模型的训练数据setting

如果要修改可视化输入图像和相机参数，则修改vis_multitask.py内的data_loader里的img_path和calib_path即可，可视化结果默认存在tmp_vis_imgs内


## 编译
在代码包根目录, 使用`projects/pilot/dev/publish/trace_compile_model.py`, Example:

        # cd to release_package root

        python3 projects/pilot/dev/publish/trace_compile_model.py \
            --sub-project galaxy_x3c_pe \
            --publish-version 0.0.1 \
            --compile-mode local \

对应参数有:

- `--sub-project`: 编译项目config名称，比如galaxy-J5项目就是galaxy_x3c_pe
- `--publish-version`: 编译模型的版本号，格式为“x.x.x"，比如"0.0.1"
- `--compile-mode`: 编译模式，本地编译为local

进行脚本测试时，为了快速完成验证，可以将编译config文件内的optimization_level设置为“O0”