# 模型生产链路
Pilot@HAT模型生产链路，自动化行为包括模型训练、模型评测，其中模型训练支持多模型、多阶段训练，支持自动resume，支持进行各类训练参数的修改。

## 环境准备
HAT相关环境构建参考[Installation Guide](../../../../docs/source/quick_start/installation.md)完成HAT基础开发环境的安装。

安装HDFlow，参考[Install HDFlow](https://gitlab.hobot.cc/auto/perception/ad/hdflow/blob/master/docs/get_start/install.md)，安装cpu版本即可。

**注意**执行时本地环境版本要求参考[requirements.txt](../../requirements.txt)

## 运行

### 方式一:

通过Shell执行

workflow example

```shell
PIPELINE=model_workflow.py
# 参考配置1：单模型
MODEL_TYPE=resize_2   # dag包含resize2单个模型
MODEL_SETTING=c385_x3c_day  # resize2模型对应setting=c385_x3c_day
MODEL_VERSION=v0.0.1

# 参考配置2：多模型，且多模型间共享model_setting
MODEL_TYPE=resize_2\;resize_4   # dag包含resize2、resize4两个模型
MODEL_SETTING=c385_x3c_day  # resize2、resize4模型均使用model_setting=c385_x3c_day
MODEL_VERSION=v0.0.1
PRETRAIN_CHECKPOINT="http://fm-meng01-wang.train.hogpu.cc/plat_gpu/pilot5_multitask_resize2_side_galaxy_x3c_side_v10-20230103_214029/output/models/pilot5_multitask_resize2_side_bayes/with_bn-checkpoint-last-840f03af.pth.tar;null"  # resize2使用http保存的pretrain ckpt、resize4不指定pretrain ckpt。该参数不支持自动多模型间share。

# 参考配置3：多模型，且多模型间分别指定model_setting
MODEL_TYPE=resize_2\;resize_4   # dag包含resize2、resize4两个模型
MODEL_SETTING=c385_x3c_day\;c385_x3c_night  # resize2模型对应setting=c385_x3c_day，resize4模型对应setting=c385_x3c_night
MODEL_VERSION=v0.0.1
PRETRAIN_CHECKPOINT="http://fm-meng01-wang.train.hogpu.cc/plat_gpu/pilot5_multitask_resize2_side_galaxy_x3c_side_v10-20230103_214029/output/models/pilot5_multitask_resize2_side_bayes/with_bn-checkpoint-last-840f03af.pth.tar;null"  # resize2使用http保存的pretrain ckpt、resize4不指定pretrain ckpt。该参数不支持自动多模型间share。


# 参考配置4：版本火车模式
MODEL_TYPE=resize_2_side_bayes  
MODEL_SETTING=galaxy_x3c_side_day_lmdb  
EVAL_DATA_SETTING=galaxy_x02_side_day\,galaxy_x03_side_day
MODEL_VERSION=v0.0.1
PRETRAIN_CHECKPOINT="http://fm-meng01-wang.train.hogpu.cc/plat_gpu/pilot5_multitask_resize2_side_galaxy_x3c_side_v10-20230103_214029/output/models/pilot5_multitask_resize2_side_bayes/with_bn-checkpoint-last-840f03af.pth.tar"  

# 参考配置5: 模型checkpoint通过resume模式加载, 继续中断的训练
MODEL_TYPE=resize_2_side_bayes  
MODEL_SETTING=galaxy_x3c_side_day_lmdb  
MODEL_VERSION=v0.0.1
PRETRAIN_CHECKPOINT="http://fm-meng01-wang.train.hogpu.cc/plat_gpu/pilot5_multitask_resize2_side_galaxy_x3c_side_v10-20230103_214029/output/models/pilot5_multitask_resize2_side_bayes/with_bn-checkpoint-last-840f03af.pth.tar"  
IS_RESUME=true  # 支持不同模型分开指定；

# 评测report相关配置
DIFF_EVAL_NAME="pilot_multitask_resize2_c385_x3c_day_test_dag_v1.2.5_release;pilot_multitask_resize2_c385_x3c_day_test_dag_v1.2.5_release"
DIFF_EVAL_NAME_RELEASE="pilot_multitask_resize2_c385_x3c_day_test_dag_v1.2.5_release;pilot_multitask_resize2_c385_x3c_day_test_dag_v1.2.5_release"


# 计算资源, num_machine x num_per_machine_gpus
TRAIN_RESOURCE=1x2
EVAL_RESOURCE=1x2

# 项目号，aidi集群提交job项目号，评测也会基于该项目号
PROJECT_ID=PDT20220001

# 集群相关配置, remote-executor时需要提供
# docker维护文档：https://horizonrobotics.feishu.cn/wiki/wikcnsicemSV1DByUBMpJ0gevMh
GPU_DOCKER=xxx
CPU_DOCKER=xxx
BRANCH=master
DAG_NAME=pilot_training_dag
QUEUE_NAME=svc-aip-cpu
# TRAIN_QUEUE=share-debug-queue-idc
TRAIN_QUEUE=idc-share-titanxp-8
EVAL_QUEUE=$TRAIN_QUEUE

python3 -c "from hdflow.cli.execute import main, parse_args;args = parse_args();main(args)" \
     --config $PIPELINE  \
     --remote-executor aidi  \
     --pipeline-name $DAG_NAME  \
     --queue-name svc-aip-cpu  \
     --project-id $PROJECT_ID  \
     --multi-model-type  $MODEL_TYPE  \
     --multi-model-setting  $MODEL_SETTING \
     --multi-eval-data-setting  $EVAL_DATA_SETTING \
     --model-version   $MODEL_VERSION  \
     --multi-model-name-suffix  $MODEL_NAME_SUFFIX  \
     --git-branch  $BRANCH  \
     --gpu-docker  $GPU_DOCKER  \
     --cpu-docker  $CPU_DOCKER  \
     --multi-train-queue  $TRAIN_QUEUE  \
     --multi-eval-queue  $EVAL_QUEUE  \
     --multi-train-resource  $TRAIN_RESOURCE  \
     --multi-eval-resource  $EVAL_RESOURCE  \
     --multi-diff-eval-name $DIFF_EVAL_NAME \
     --multi-diff-eval-name-release $DIFF_EVAL_NAME_RELEASE \
     --enable-eval  \
     --enable-auto-threshold \
     --enable-tracking  \
     --rerun-with-resume  \
     --pipeline-test \
     #  --multi-model-name-suffix  xxx  \
     #  --multi-eval-name-suffix  xxx  \
     #  --multi-pretrain-checkpoint $PRETRAIN_CHECKPOINT  \
     #  --multi-resume-mode $IS_RESUME \
```
说明：

模型生产链路支持通过一个DAG提交多个模型的训练、评测。因此在支持的参数中经常需要指定不同模型的参数。
例如对于训练模型使用的集群，模型A使用A集群，模型B可能使用B集群。我们对于此类需要支持多个模型指定不同配置的参数，以`multi-`开头来标记。
对于带`multi-`前缀的参数，当需要对每个模型分别指定不同配置时，统一使用`;`来进行参数分割，例如若需要训练resize_2，resize_4两个模型，
可以通过指定`multi_model_type=resize_2;resize_4`来设置两个需要生产的模型。其他`multi-`参数，当使用`;`来指定不同模型参数时，其参数与`multi_model_type`一一对应；
例如`multi_model_type=resize_2;resize_4`，当需要resize_2模型指定`model_setting=day`，resize_4模型指定`model_setting=night`时，
我们设置`multi_model_setting=day;night`即可。
当启用版本火车时，想要评测火车头下的所有ltc，则需要设置multi-eval-data-setting，例如`EVAL_DATA_SETTING=c385_x3c_day\,c673_x3c_day\;c385_x3c_night\,c673_x3c_night`,";"来分割不同的模型，","分割同一火车头下不同的Ltc。  

**注意**在shell脚本中，`;`前通常需要加一个转义符`\`。

目前，以下参数支持使用`;`来进行参数分割，并分别指定：

* multi-model-type
* multi-model-setting
* multi-eval-data-setting
* multi-model-name-suffix
* multi-eval-name-suffix
* multi-diff-eval-name
* multi-diff-eval-name-release
* multi-pretrain-name
* multi-pretrain-version
* multi-pretrain-stage
* multi-pretrain-checkpoint
* multi-resume-mode
* multi-train-queue
* multi-eval-queue
* multi-train-resource
* multi-eval-resource
* multi-train-stages

在某些情况下，我们希望参数在模型间共享，省去单独设置的麻烦，这时只需指定单个参数即可，脚本会自动在多模型间share参数。
例如当`multi_model_type=resize_2;resize_4`时，需要生产resize_2、resize_4两个模型，此时若希望训练集群统一为xxx，则指定`multi_train_queue=xxx`即可。
目前，以下参数支持自动share参数：

* multi-model-setting
* multi-model-name-suffix
* multi-eval-name-suffix
* multi-train-queue
* multi-eval-queue
* multi-train-resource
* multi-eval-resource
* multi-resume-mode

另外，一些模型相关参数可以省略，此时脚本会前往[model_meta.yaml](../../model_meta.yaml)中寻找相对应参数作为默认参数。
包括:

* multi-train-stages
* multi-train-resource
* multi-eval-resource

参数说明：

* config: 必填，workflow的定义文件路径，以pilot为例，存放在[here](model_workflow.py)。
* local-executor/remote-executor: 必填， 执行时指定的executor，支持本地或aidi执行。
* multi-model-type: 必填，模型type，通常对应[model_meta.yaml](../../model_meta.yaml)中type，该参数也用来确定生产链路生产模型的数量。
* multi-model-setting: 必填，模型setting。
* multi-eval-data-setting: 版本火车模式下必填，模型评测集数据setting。
* model-version: 必填，模型version，需要严格遵循`vx.x.x`的格式，**注意**该参数所有模型共享。
* multi-model-name-suffix: 选填，模型后缀，影响训练任务命名，以及enable_tracking的情况下上传模型的名称。
* multi-eval-name-suffix: 选填，enable_eval=True时必填，评测prediction后缀，影响AIDI评测中leaderboard上的名称。
* multi-diff-eval-name: 选填，enable_auto_threshold=True时必填，生产report时需要指定的prediction名(即AIDI评测中leaderboard上的名称)，通常为卡阈值前结果。
* multi-diff-eval-name-release：选填，enable_auto_threshold=True时必填，生产report时需要指定的prediction名(即AIDI model eval中leaderboard上的名称)，通常为卡阈值后结果。
* multi-pretrain-name/version/stage: 选填，指定模型训练使用的pretrain模型，通过`aidi://{pretrain_name}/{pretrain_version}/{pretrain_stage}`的形式来读取ckpt。`Note:`若其中某个模型不需要pretrain，通过设置`null`来跳过该模型，此时name/version/stage都需要指定为`null`。
* multi-pretrain-checkpoint: 选填，指定模型训练使用的pretrain模型，通过url的形式来读取ckpt。`Note:`若其中某个模型不需要pretrain，通过设置`null`来跳过该模型。
* multi-resume-mode: 选填，指定模型加载pretrain_checkpoint时的方式，若为True，则通过resume的方式加载，会根据ckpt来恢复step以及optimizer的状态；否则作为pretrain重新开始训练。
```
特别注意！！!
对于某一个模型，当multi-resume-mode和rerun-with-resume同时开启时，若期望既能通过resume的方式加载checkpoint，而且还能支持训练失败后能够自动resume训练，我们需要确保在该版本号下没有运行过DAG
(若之前有运行记录，可以调整版本号)；
否则，dag不会以指定的ckpt来加载训练状态！
```
* git-branch/git-commit-id/git-tag: 选填(当集群训练时必填)。环境代码，生产链路基于该代码来运行，支持选择HAT git branch、commit id、tag。**注意**本地执行DAG时，该参数无效，基于当前工作目录执行。
* gpu(cpu)-docker: 选填(当集群训练时必填)。生产链路使用的docker环境。需要注意，部分DAG节点需要cpu_docker，且通常不能与gpu_docker混用。当未提供该参数时，使用[pilot_cluster_cfg](../pilot_cluster_cfg.yaml)中的`docker_image`配置。参考：Pilot [docker维护文档](https://horizonrobotics.feishu.cn/wiki/wikcnsicemSV1DByUBMpJ0gevMh)
* multi-train-resource: 选填(缺省时使用[model_meta.yaml](../../model_meta.yaml)中对应resource)。可json解析的字符串，格式参考：{\"num_machines\":1,  \"num_gpus_per_machine\": 2} 或者 [{\"num_machines\":1,  \"num_gpus_per_machine\": 2}, {\"num_machines\":1,  \"num_gpus_per_machine\": 2}]，后者为list，指定不同模型的训练资源。
* multi-eval-resource: 参考multi-train-resource。
* disable-train: 是否关闭模型训练。default=False
* enable-eval: 是否执行模型评测。default=False
* enable-auto-threshold: 是否执行自动卡阈值评测。default=False
* enable-tracking: 是否开启tracking。default=False
* rerun-with-resume: 是否开启自动resume。default=False
* pipeline-test: 是否pipeline-test。default=False
* upload-metric-to-doris：是否上传本次评测报告指标。default=False

AIDI集群相关参数：

* pipeline-name: 必填，aidi前端job名。
* queue-name: 必填，DAG执行集群，通常为CPU集群。
* project-id: 选填(enable-eval=True时必填)，aidi集群提交job项目号，评测也会基于该项目号。
* multi-train-queue: 必填，模型训练集群，通常为GPU集群。支持多模型单独指定。
* multi-eval-queue: 必填，模型评测集群，通常为GPU集群。支持多模型单独指定。

### 方式二(推荐)

相对于方式一通过shell脚本来提交DAG任务，我们还提供了前端web的方式来提交DAG。

安装前端依赖[streamlit](https://streamlit.io/)：
```shell
pip3 install streamlit
```

本地开发机启动前端服务:
```shell
streamlit run projects/pilot/tools/production/app.py
```
运行该命令行，会返回一个URL，如下：
```log
2023-04-10 18:15:42.310 Did not auto detect external IP.
Please go to https://docs.streamlit.io/ for debugging hints.

  You can now view your Streamlit app in your browser.

  Network URL: http://10.10.112.106:8502
```
复制http://10.10.112.106:8502并打开游览器进入该页面，即可通过前端提交DAG。

在前端中，我们能够设置全局配置，如DAG包含的节点、DAG名称等等。同时，也能够灵活的配置多个待生产模型。
