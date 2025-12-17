# person
python projects/mono/person_cyclist_detection/eval/evaluate.py \
    --gt-file "/horizon-bucket/auto_eval/adas_eval/eval_platform/fs/6033372/datasets/x8b_20220412_cp/data.json" \
    --setting-file "/horizon-bucket/auto_eval/adas_eval/eval_platform/fs/6033372/settings/7366440/Merge_Ped_filter_Day_0-90m.yaml" \
    --predict-file "kPed2PE.json"\
    --save-dir "output"

# cyclist 
python projects/mono/person_cyclist_detection/eval/evaluate.py \
    --gt-file "/horizon-bucket/auto_eval/adas_eval/eval_platform/fs/6033428/datasets/x8b_20220412_cp/data.json" \
    --setting-file "/horizon-bucket/auto_eval/adas_eval/eval_platform/fs/6033428/settings/7366558/Merge_Cyc_filter_Day_0-90m.yaml" \
    --predict-file "kCyc2PE.json"\
    --save-dir "output"