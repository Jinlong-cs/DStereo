# offline pack ReadMe

## 使用说明
直接运行workflow_detection_offline.py，指定以下参数：   
workflow-task-name --------- task-name，对齐batch_runner.py   
workflow-annoset-name --------- annoset-name，对齐batch_runner.py    
workflow-purpose --------- train，离线打包当前只支持train    
--output-root --------- 本地打包输出位置     
workflow-source-data-root-url  --------- 本地data_v数据存储位置, 如有多个输入，用空格隔开       

## commend demo   
cd examples    
python3 -u mono/data_management_and_deploy/densebox/workflow_detection_offline.py \
  --workflow-annoset-name 2pe_vehicle_rear_fp --workflow-task-name 2pe_vehicle_rear_fp \
  --workflow-tags test --workflow-purpose train \
  --workflow-num-workers 8 --workflow-day-or-night all \
  --workflow-pass-invalid --workflow-trans-anno-to-pbrec \
  --allow-wo-inputs \
  --output-root "" \
  --workflow-source-data-root-url /jfs-public/adas/yue01.shi/fp/data/x8b/vehicle_rear_fp/pack_20220727/20220726/60695 /jfs-public/adas/yue01.shi/fp/data/x8b/vehicle_rear_fp/pack_20220727/20220726/60632
