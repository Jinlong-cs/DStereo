# export CUDA_VISIBLE_DEVICES=0,1

hpflow_run evaluate \
    --data projects/mono/person_cyclist_detection_4cls/infer/config/datasets.py \
    --config projects/mono/person_cyclist_detection_4cls/infer/config/hat_2pe_person_detection.py \
    --ctx 0,1 --num-workers 4 --overwrite --upload --eval-task kPed2PE  --project-id PDT20220004

hpflow_run evaluate \
    --data projects/mono/person_cyclist_detection_4cls/infer/config/datasets.py \
    --config projects/mono/person_cyclist_detection_4cls/infer/config/hat_2pe_person_detection.py \
    --ctx 0,1 --num-workers 4 --overwrite --upload --eval-task kCyc2PE  --project-id PDT20220004

hpflow_run evaluate \
    --data projects/mono/person_cyclist_detection_4cls/infer/config/datasets.py \
    --config projects/mono/person_cyclist_detection_4cls/infer/config/hat_2pe_person_detection.py \
    --ctx 0,1 --num-workers 4 --overwrite --upload --eval-task kMotorCyc2PE  --project-id PDT20220004

hpflow_run evaluate \
    --data projects/mono/person_cyclist_detection_4cls/infer/config/datasets.py \
    --config projects/mono/person_cyclist_detection_4cls/infer/config/hat_2pe_person_detection.py \
    --ctx 0,1 --num-workers 4 --overwrite --upload --eval-task kTriCyc2PE  --project-id PDT20220004