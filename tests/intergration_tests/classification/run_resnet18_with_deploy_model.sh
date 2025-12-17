#!/usr/bin/env bash

set -e

cfg=tests/data/resnet18_with_deploy_model.py

sed -i 's/enable_model_tracking = False/enable_model_tracking = True/g' ${cfg}

# float train and predict
python3 tools/train.py -s float -c ${cfg} $@
python3 tools/predict.py -s float -c ${cfg} $@

# calibration train
python3 tools/train.py -s calibration -c ${cfg} $@

# model checker
python3 tools/deploy/model_checker.py -c ${cfg} $@

# qat train and predict
python3 tools/train.py -s qat -c ${cfg} $@
python3 tools/predict.py -s qat -c ${cfg} $@

# int_infer predict
python3 tools/predict.py -s int_infer -c ${cfg} $@

# test debug tools
# featuremap similarity
python3 tools/analyze/model_profiler.py -c ${cfg} $@

# compare weights
sed -i 's/tool=featuremap_similarity_tool/tool=compare_weights_tool/g' ${cfg}
python3 tools/analyze/model_profiler.py -c ${cfg} $@

# model profiler
sed -i 's/tool=compare_weights_tool/tool=model_profiler_tool/g' ${cfg}
python3 tools/analyze/model_profiler.py -c ${cfg} $@

# profile feauremap
sed -i 's/model_convert_pipeline=similarity_convert_pipeline/model_convert_pipeline=other_convert_pipeline/g' ${cfg}
sed -i 's/tool=model_profiler_tool/tool=profile_featuremap_tool/g' ${cfg}
python3 tools/analyze/model_profiler.py -c ${cfg} $@

# check shared
sed -i 's/tool=profile_featuremap_tool/tool=check_shared_tool/g' ${cfg}
python3 tools/analyze/model_profiler.py -c ${cfg} $@

# check fused
sed -i 's/tool=check_shared_tool/tool=check_fused_tool/g' ${cfg}
python3 tools/analyze/model_profiler.py -c ${cfg} $@

# check qconfig
sed -i 's/model_convert_pipeline=other_convert_pipeline/model_convert_pipeline=fx_pipeline/g' ${cfg}
sed -i 's/tool=check_fused_tool/tool=check_qconfig_tool/g' ${cfg}
python3 tools/analyze/model_profiler.py -c ${cfg} $@

# check deploy device
sed -i 's/tool=check_qconfig_tool/tool=deploy_device_tool/g' ${cfg}
python3 tools/analyze/model_profiler.py -c ${cfg} $@

# resume origin cfg
sed -i 's/model_convert_pipeline=fx_pipeline/model_convert_pipeline=similarity_convert_pipeline/g' ${cfg}
sed -i 's/tool=deploy_device_tool/tool=featuremap_similarity_tool/g' ${cfg}

# export onnx
# float
python3 tools/deploy/export_onnx.py -c ${cfg} $@

sed -i 's/stage="float"/stage="qat"/g' ${cfg}
sed -i 's/model_convert_pipeline=float_predictor\["model_convert_pipeline"]/model_convert_pipeline=qat_predictor\["model_convert_pipeline"]/g' ${cfg}
python3 tools/deploy/export_onnx.py -c ${cfg} $@

sed -i 's/stage="qat"/stage="int_infer"/g' ${cfg}
sed -i 's/model_convert_pipeline=qat_predictor\["model_convert_pipeline"]/model_convert_pipeline=int_infer_predictor\["model_convert_pipeline"]/g' ${cfg}
python3 tools/deploy/export_onnx.py -c ${cfg} $@

# resume
sed -i 's/stage="int_infer"/stage="float"/g' ${cfg}
sed -i 's/model_convert_pipeline=int_infer_predictor\["model_convert_pipeline"]/model_convert_pipeline=float_predictor\["model_convert_pipeline"]/g' ${cfg}

send=`date '+%Y-%m-%d %H:%M:%S'`
echo $send

# test compile for BAYES BPU

# Temporarily commented out dut to unknow bug of horizon-plugin-profiler in dcu machine
# TODO, kaiwen.kong: Figure out horizon-plugin-profiler bug and fix.
if [ "${USE_DCU}" != "1" ]; then
cat ${cfg} > cfg_bayes.py
echo "--- start compile for BAYES BPU ---"
python3 tools/deploy/compile_perf.py -c cfg_bayes.py $@

# test compile for BERNOULLI2 BPU
sed "s/BAYES/BERNOULLI2/g" cfg_bayes.py > cfg_bernoulli2.py
echo "--- start compile for BERNOULLI2 BPU ---"
python3 tools/deploy/compile_perf.py -c cfg_bernoulli2.py $@

rm cfg_bayes.py cfg_bernoulli2.py
fi

python3 tools/analyze/perf_dataloader.py --config ${cfg} --stage float --allow-all-rank -ids "0"
python3 tools/analyze/perf_model_training.py --config ${cfg} --stage float --allow-all-rank -ids "0"


send=`date '+%Y-%m-%d %H:%M:%S'`
echo $send

# hybrid model test
sed -i 's/dict(type="Float2QAT")/dict(type="Float2QAT", convert_mode="fx", hybrid=True)/g' ${cfg}
sed -i 's/dict(type="Float2Calibration")/dict(type="Float2Calibration", convert_mode="fx", hybrid=True)/g' ${cfg}
sed -i 's/dict(type="QAT2Quantize")/dict(type="QAT2Quantize", convert_mode="fx")/g' ${cfg}
# calibration train
python3 tools/train.py -s calibration -c ${cfg} $@

# qat train and predict
python3 tools/train.py -s qat -c ${cfg} $@
python3 tools/predict.py -s qat -c ${cfg} $@

# int_infer predict
python3 tools/predict.py -s int_infer -c ${cfg} $@

# export onnx
sed -i 's/stage="float"/stage="qat"/g' ${cfg}
sed -i 's/model_convert_pipeline=float_predictor\["model_convert_pipeline"]/model_convert_pipeline=qat_predictor\["model_convert_pipeline"]/g' ${cfg}
python3 tools/deploy/export_onnx.py -c ${cfg} $@

# resume
sed -i 's/dict(type="Float2QAT", convert_mode="fx", hybrid=True)/dict(type="Float2QAT")/g' ${cfg}
sed -i 's/dict(type="Float2Calibration", convert_mode="fx", hybrid=True)/dict(type="Float2Calibration")/g' ${cfg}
sed -i 's/dict(type="QAT2Quantize", convert_mode="fx")/dict(type="QAT2Quantize")/g' ${cfg}
sed -i 's/stage="qat"/stage="float"/g' ${cfg}
sed -i 's/model_convert_pipeline=qat_predictor\["model_convert_pipeline"]/model_convert_pipeline=float_predictor\["model_convert_pipeline"]/g' ${cfg}

# quant analysis test
python3 tools/analyze/quant_analysis.py -c ${cfg} $@

# sensitive op qconfig setter test
sed -i 's/sensitive_op_qconfig_test = False/sensitive_op_qconfig_test = True/g' ${cfg}
python3 tools/train.py -s calibration -c ${cfg} $@

# resume
sed -i 's/sensitive_op_qconfig_test = True/sensitive_op_qconfig_test = False/g' ${cfg}

# export qat hbir
python3 tools/deploy/export_hbir.py -c ${cfg}

# run hbir model profiler
sed -i 's/tool=featuremap_similarity_tool/tool=hbir_profiler_tool/g' ${cfg}
sed -i 's/model_convert_pipeline=similarity_convert_pipeline/model_convert_pipeline=hbir_pipeline/g' ${cfg}
python3 tools/analyze/model_profiler.py -c ${cfg} $@

# run model profilerv2
sed -i 's/tool=hbir_profiler_tool/tool=model_profiler_v2_tool/g' ${cfg}
sed -i 's/model_convert_pipeline=hbir_pipeline/model_convert_pipeline=fx_pipeline/g' ${cfg}
python3 tools/analyze/model_profiler.py -c ${cfg} $@

# resume
sed -i 's/tool=model_profiler_v2_tool/tool=featuremap_similarity_tool/g' ${cfg}
sed -i 's/model_convert_pipeline=fx_pipeline/model_convert_pipeline=similarity_convert_pipeline/g' ${cfg}


send=`date '+%Y-%m-%d %H:%M:%S'`
echo $send
