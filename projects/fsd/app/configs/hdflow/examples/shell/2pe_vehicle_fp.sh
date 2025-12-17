cd ../

# export TEST_MODE=1
conda activate hdflow
# python -u mono/data_management_and_deploy/batch_runner.py

python3 -u mono/data_management_and_deploy/densebox/workflow_detection_offline.py \
  --workflow-annoset-name 2pe_vehicle_rear_fp --workflow-task-name 2pe_vehicle_rear_fp \
  --workflow-tags test --workflow-purpose train \
  --workflow-num-workers 8 --workflow-day-or-night all \
  --workflow-pass-invalid --workflow-trans-anno-to-pbrec \
  --allow-wo-inputs \
  --output-root "" \
  --workflow-source-data-root-url /jfs-public/adas/yue01.shi/fp/data/x8b/vehicle_rear_fp/pack_20220727/20220726/60695 /jfs-public/adas/yue01.shi/fp/data/x8b/vehicle_rear_fp/pack_20220727/20220726/60632

