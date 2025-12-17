export CUDA_VISIBLE_DEVICES=2

hpflow_run evaluate \
    --data projects/mono/person_cyclist_detection_4cls/infer/config/datasets.py \
    --config projects/mono/person_cyclist_detection_4cls/infer/config/hat_2pe_person_detection.py \
    --ctx 0 --num-workers 2 --overwrite --eval-task kPed2PE  --project-id PDT20220004 --eval-num 100

hpflow_run evaluate \
    --data projects/mono/person_cyclist_detection_4cls/infer/config/datasets.py \
    --config projects/mono/person_cyclist_detection_4cls/infer/config/hat_2pe_person_detection.py \
    --ctx 0,1 --num-workers 4 --overwrite --eval-task kCyc2PE  --project-id PDT20220004

hpflow_run evaluate \
    --data projects/mono/person_cyclist_detection_4cls/infer/config/datasets.py \
    --config projects/mono/person_cyclist_detection_4cls/infer/config/hat_2pe_person_detection.py \
    --ctx 0,1 --num-workers 4 --overwrite --eval-task kMotorCyc2PE  --project-id PDT20220004

hpflow_run evaluate \
    --data projects/mono/person_cyclist_detection_4cls/infer/config/datasets.py \
    --config projects/mono/person_cyclist_detection_4cls/infer/config/hat_2pe_person_detection.py \
    --ctx 0,1 --num-workers 4 --overwrite --eval-task kTriCyc2PE  --project-id PDT20220004