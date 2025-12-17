python tools/train.py \
    --stage int_infer \
    --config projects/mono/person_cyclist_detection_4cls/multitask.py

# J5
python tools/deploy/compile_standalone.py \
    ./tmp_output/mono_multitask_2pe_person/int_infer-deploy-checkpoint-last.pt \
    --input-size 1x3x128x128 \
    --output-layout NHWC \
    --opt O3  \
    --march bayes \
    --name person_cyclist_2pe_multitask \
    --input-source resizer \
    --extra-args "--dev-remove-extra-output-cpu-op --max-time-per-fc 1000" \
    --output tmp_compile_person_cyclist_2pe_multitask_j5 

# J3
python tools/deploy/compile_standalone.py \
    ./tmp_output/mono_multitask_2pe_person/int_infer-deploy-checkpoint-last.pt \
    --input-size 1x3x128x128 \
    --output-layout BPU_RAW \
    --opt O3  \
    --march bernoulli2 \
    --name person_cyclist_2pe_multitask \
    --input-source resizer \
    --extra-args "--dev-remove-extra-output-cpu-op --max-time-per-fc 1000" \
    --output tmp_compile_person_cyclist_2pe_multitask_j3