(aidi_eval/how_to_use_aidi_eval)=
# 如何使用 AIDI_EVAL

`HAT` 目前支持上传模型预测结果到 AIDI 评测系统进行评测，主要依靠 `AIDIEval callbacks` 和 `Predictor` 来完成评测功能。


## 基本使用

使用示例如下：


```python
# 由于目前HAT预测框架还尚未完善，许多模块需要由用户自己处理

# 1. 对要 batch_data 进行处理
# 可以自定义函数，对即将喂给模型的 batch_data 进行处理，比如过滤字段等
def filter_input_batch(batch):
    batch_data = batch
    ...
    pass
    

# 2. 根据 batch_data 和 model_outs 生成预测结果
# 在这个方法里，用户可以自由操作，只需要最后满足格式的预测结果即可
# （但用户也要自己保证生成的结果，可以在 AIDI 平台正常评测）

def reformat_aidi_eval_out(batch, model_outs, task_name):
    
    # 2.1 (可选)，可对 batch, model_outs 进行处理
    # batch = reformat_batch_func(batch)
    # model_outs = reformat_model_outs_func(model_outs)
    
    
    """ 需要生成的格式如下:
    
    [ # 最外层是个 List
        
        { # 每个 dict，是一张图片的预测结果，这是第一张
            "image_key": 00001.png,
            task_name: [
                {}, {}   # 这个里面每个 dict，是一个 bbox 和 对应的score
            ]
        },
        
        { # 每个 dict，是一张图片的预测结果，这是第二张
            "image_key": 00002.png,
            task_name: [
                {}, {}
            ]
        },

    ]

    """

    final_results = []
    for data_i, outs_i in zip(batch, model_outs):
        image_key = data_i["img_key"],
            
        one_img_preds  = []  # 一张图片中的预测
        for one_pred in outs_i:
            one_pred_dict = {
                "bbox": one_pred[:4],  # bbox prediction
                "bbox_score": float(out_ij[5]), # score prediction
            }
            one_img_preds.append(one_pred_dict)
                
        data_i_prediction = {
            "image_key": image_key,
            str(task_name): one_img_preds,
        }
        
        final_results.append(data_i_prediction)
        
    return final_results
        

        
# 3. 定义 AIDIEval callbacks
aidi_eval_callback = dict(
    type="AIDIEval",
    project_id="123456",  # 你的 ProjectID
    prediction_name="test202111231900",  # 自定义的 Prediction Name 
    prediction_tags=["merge"],
    aidi_eval_dataset_id=6026304,  # Dataset ID, int 类型
    aidi_eval_task_type=AIDIEvalTaskType.DET,  # AIDIEvalTaskType，目前仅支持 Detection2D
    reformat_output_fn=reformat_aidi_eval_out,
    reformat_input_fn=filter_input_batch,
    task_name=task_name,  # 预测的 task_name，比如 "person", "vehicle" 等，会传给 reformat_output_fn
)

# 以上内容就是关于 AIDIEval callback 的使用，也可以
# 4. 定义 Predictor
num_processes = 1
predictor = dict(
    type="Predictor",
    model=model,
    data_loader=val_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=None,
    callbacks=aidi_eval_callback,
    log_interval=50,
)

predict_solver = dict(
    predictor=predictor,
    train_step="qat",
    checkpoint=os.path.join(ckpt_dir, "/qat-checkpoint-best.pth.tar"),
    ckpt_step="qat",
    ignore_extra=True,
)

```

以上内容即是使用 HAT `Predictor` +  `AIDIEval` 进行预测的基本设置，然后只需执行以下命令即可：
```bash
python3 tools/predictor --config ...
```


## FAQ

**Q1. 如何在训练 pipeline 中提交每个epoch的预测结果到 AIDI 平台进行评测？**

**A:**  `AIDIEval` 是一个 `callback`，也可以脱离 HAT 预测框架独立使用，把它当成一个普通的 `callback` 放入到训练 pipeline 的 validation_callbacks 中即可。
   
**Q2. 预测时如果是多个 dataloader 时怎么处理？**

**A:** 当前的 `AIDIEval callback` 仅支持对单一 dataset 和单一 dataloader 的处理，多个 dataloader 时，需要对应有多个 `AIDIEval callback`。例如:
   ```python
    def create_loaders_and_callbacks(multi_task_loader, **kwargs):
        """自定义方法，生成 dataloader list 和 callback list"""
        val_lodar_list, aidi_eval_callback_list = [], []
        
        for task_name, loader in multi_task_loader["loaders"].items():
            val_lodar_list.append(loader)
            
            # 对每个 dataset(task_name)，生成对应的 AIDIEval callbacks
            callbacks = dict(
                type="AIDIEval",
                output_root="./example-pred-outs/",
                project_id="TD2020006",
                prediction_name="example-prediction-test",
                prediction_tags=EVAL_INFO.adas_eval_predict_tags,
                aidi_eval_dataset_id=DATASET_ID[task_name],
                aidi_eval_task_type=AIDIEvalTaskType.Det2D,
                reformat_output_fn=reformat_aidi_eval_out,
                reformat_input_fn=filter_input_batch,
                task_name=task_name,
            )
            aidi_eval_callback_list.append(callbacks)

        # 最后返回对应的 data_loader list 和 callback list
        return val_loader_list, aidi_eval_callback_list
    loaders_list, callbacks_list = create_loaders_and_callbacks()
    
    predictor = dict(
        type="Predictor",
        model=test_model,
        data_loader=loaders_list,  # dataloader list
        batch_processor=val_batch_processor,
        device=None,
        metrics=None,
        callbacks=callbacks_list,  # callbacks list
        log_interval=50,
        share_callbacks=False,  # 这里要设置 share_callbacks = False
    )
   ```
   
   