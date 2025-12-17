1.训练

1.1 指定j3/j5
j3: projects/mono/person_cyclist_detection_4cls/common.py指定  march = March.BERNOULLI2   input_shift = 6 
j5: projects/mono/person_cyclist_detection_4cls/common.py指定  march = March.BAYES  input_shift = 12 

* j3 int阶段使用dpp不支持shift6，需要在horizon_plugin_pytorch/nn/qat/detection_post_process_v1.py和horizon_plugin_pytorch/nn/quantisized/detection_post_process_v1.py中将shift修改为6

1.2 执行训练
参考projects/mono/person_cyclist_detection_4cls/train_run.sh
float: python tools/train.py --stage float --config projects/mono/person_cyclist_detection_4cls/multitask.py --pipeline-test --device-ids 0
qat:   python tools/train.py --stage qat --config projects/mono/person_cyclist_detection_4cls/multitask.py --pipeline-test --device-ids 0

参考projects/mono/person_cyclist_detection_4cls/compile.sh
int-infer: python tools/train.py --stage int_infer --config projects/mono/person_cyclist_detection_4cls/multitask.py

2.编译
参考projects/mono/person_cyclist_detection_4cls/compile.sh
j3: 
python tools/deploy/compile_standalone.py \
    ./tmp_output/mono_multitask_2pe_person/int_infer-deploy-checkpoint-last.pt \
    --input-size 1x3x128x128 \
    --output-layout BPU_RAW \
    --opt O3  \
    --march bernoulli2 \
    --name person_cyclist_2pe_multitask \
    --input-source resizer \
    --extra-args "--dev-remove-extra-output-cpu-op --max-time-per-fc 1000" \
    --output tmp_compile_person_cyclist_2pe_multitask_v2.2_j3

j5:
python tools/deploy/compile_standalone.py \
    ./tmp_output/mono_multitask_2pe_person/int_infer-deploy-checkpoint-last.pt \
    --input-size 1x3x128x128 \
    --output-layout NHWC \
    --opt O3  \
    --march bayes \
    --name person_cyclist_2pe_multitask \
    --input-source resizer \
    --extra-args "--dev-remove-extra-output-cpu-op --max-time-per-fc 1000" \
    --output tmp_compile_person_cyclist_2pe_multitask_j5_v4_tmp 

3.评测
执行projects/mono/person_cyclist_detection_4cls/infer/infer_model_local.sh

* 如果环境中没有hpflow，需要将本地hpflow的环境变量加入export PYTHONPATH=$PYTHONPATH:hpflowpath
* j5评测，由于使用dpp进行评测，需要修改rpn_out_keys=["pred_boxes_out", "pred_scores", "pred_cls"]
*     在horizon_plugin_pytorch/nn/qat/detection_post_process_v1.py和horizon_plugin_pytorch/nn/quantisized/detection_post_process_v1.py中将shift修改为12 score的output数据类型改为int16