# PackInfer

## 本地PackInfer

```
python3 tools/predict.py \
        --config projects/pilot/configs/{MODEL_TYPE}/pack_infer_multitask.py \
        --stage pack_infer \
        --hat-pilot-model-setting {MODEL_SETTING} \
        --hat-pilot-model-checkpoint {MODEL_CKPT} \
        --hat-pilot-multiview-pack {MULTIVIEW_PACK} \
        --hat-pilot-bev-pack {BEV_PACK} \
        --hat-pilot-homo-offset {HOMO_OFFSET} \
        --hat-pilot-temporal-hommo-offset {TEMPORAL_HOMO_OFFSET} \
        --hat-pilot-model-thresh {MODEL_THRESH} \
```
部分参数说明如下：
* `MODEL_TYPE`:【必需】 模型类型,如bev_7v；目前稳定支持的只有BEV模型(bev_7v, bev_7v_temporal, bev_5v)
* `--stage`: 【必须】使用`pack_infer`这个stage
* `--hat-pilot-model-setting`: 【必须】 模型配置名；如：pilot5.1_master、ek_bev;. 参见[BEVModelSetting](../../configs/project_common/enum.py)
* `--hat-pilot-model-checkpoint`: 【必须】 模型参数地址，必须是QAT阶段的结果。
* `--hat-pilot-pack-infer-consist`： Packinfer是否做一致性对齐。是, 则传参如下:`--hat-pilot-pack-infer-consist "1"`
* `--hat-pilot-pack-infer-vis`: PackInfer是否做可视化。是, 则传参如下:`--hat-pilot-pack-infer-vis "1"`
* `--hat-pilot-multiview-pack`: 【必须】 多视角Pack路径的通配符；路径配置通配符示例如下：`/BUCKET_ROOT/DIR/ADAS_20221119-155359_314_$Index.pack`
* `--hat-pilot-bev-pack`: 【可选】BEV的topic对应pack；可用来作为同步帧的控制信息。实现按照该Pack完成帧同步后做Infer。
* `--hat-pilot-homo-offset`: 【一致性必须】 各视角的homo_offset存储路径。**目前一致性对齐必须提供(软件同学回灌后dump的结果)**
* `--hat-pilot-temporal-hommo-offset`: 【一致性必须】 时序特征homo_offset存储路径。**目前一致性对齐必须提供(软件同学回灌后dump的结果)**
* `--hat-pilot-model-thresh`: 【参数可选，变量必须】 模型的具体参数。两种传参方式，面向不同的使用需求. **PackInfer必须提供，防止配置与模型包不一致**
  * 代码修改：不使用该参数传参，直接在代码中修改；适用于本地packinfer快速查看。在[project_common](../../configs/project_common.py)中直接指定。示例如下：
  ```
  # model_thresh = os.getenv("HAT_PILOT_MODEL_THRESH")
  model_thresh = json.dumps(
        dict(
                bev_3d_vehicle=dict(
                score_threshold=0.15,
                roi_score_threshold=[0.15] * 7,
                ),
                bev_3d_vrumerge=dict(
                score_threshold=[0.17, 0.19],
                roi_score_threshold=[0.17, 0.19],
                use_maxpool=False,
                ),
                online_mapping=dict(
                lane=0.57,
                roadedge=0.45,
                ),
                bev_arrow=dict(
                score_threshold=0.2,
                iou_threshold=0.2,
                ),
                bev_junction=dict(
                score_threshold=0.2,
                iou_threshold=0.2,
                ),
                bev_roadmarking=dict(
                score_threshold=0.2,
                iou_threshold=0.2,
                ),
        ),
  )
  ```
  * 跨平台传参：用json序列阈值参数化后传参即可。目的在于支持跨平台调用,如下  `--hat-pilot-model-thresh json.dumps({MODEL_THRESH})`

## 集群PackInfer
```
python3 projects/pilot/tools/pack_infer/pilot_cluster_pack_infer.py \
    --model-type MODEL_TYPE \
    --model-setting MODEL_SETTING \
    --model-version MODEL_VERSION \
    --project-id PROJECT_ID \
    --current-cluster CLUSTER \
    --num-machines NUM_MACHINES \
    --num-gpus-per-machine NUM_GPUS_PER_MACHINE \
    --model-name-postfix NAME_POSTFIX \
    --load-publish \
    --model-checkpoint MODEL_CKPT \
    --model-thresh json.dumps(MODEL_THRESH) \
    --pack-consist \
    --pack-viz \
    --multiview-pack MULTIVIEW_PACK \
    --bev-pack BEV_PACK \
    --homo-offset HOMO_OFFSET \
    --temporal-hommo-offset TEMPORAL_HOMO_OFFSET \
    --to-video \
    --local \
    --pipeline-test
```

部分参数说明如下(与本地一致的参数不在赘述)：
 * `--model-type`: 【必选】同本地参数
 * `--model-setting`: 【必选】同本地参数
 * `--model-version`: 【必选】模型版本号
 * `--project-id`: 【必选】项目号
 * `--current-cluster`: 【必选】执行集群
 * `--num-machines`: 【可选】目前支持单机单卡
 * `--num-gpus-per-machine`: 【可选】目前支持单机单卡
 * `--model-name-postfix`: 【可选】模型名称后缀
 * `--load-publish`: 【条件可选】如果开启，会根据model_type和model_setting，自动加载当前分支中发布模型的model_checkpoint和model_thresh。[编译配置文件](../../dev/publish/cfg)。
 * `--model-checkpoint`: 【参数可选，变量必须】如果开启`--load-publish`，会自动加载模型参数地址，否则请在此处传参。
 * `--model-thresh`: 【参数可选，变量必须】如果开启`--load-publish`，会自动加载模型阈值，否则请在此处传参。
 * `--pack-consist`: 【可选】是否执行一致性对齐的关键结果dump.
 * `--pack-viz`: 【可选】是否开启packinfer的可视化。
 * `--multiview-pack`: 【可选】使用的pack地址。路径配置通配符示例如下：`/BUCKET_ROOT/DIR/ADAS_20221119-155359_314_$Index.pack`
 * `--bev-pack`: 【可选】BEV的topic对应pack；可用来作为同步帧的控制信息。实现按照该Pack完成帧同步后做Infer。
 * `--homo-offset`: 【可选】同本地参数
 * `--temporal-hommo-offset`: 【可选】同本地参数
 * `--to-video`: 【可选】是否把可视化图片转换为视频；结果保存到packinfer_video目录下。注意：生成视频后会删除原图片。
 * `--local`: 【可选】是否在本地执行。也可像前文一样直接本地传参使用
 * `--pipeline-test`: 【可选】是否链路测试。会影响Packinfer的做大Frame量
