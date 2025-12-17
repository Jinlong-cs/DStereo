# HPP
## HPP训练
```
# Float stage
python3 tools/train.py --stage float --config projects/mono/holistic_path_prediction/holistic_path_prediction.py --device-ids ...
# QAT stage
python3 tools/train.py --stage qat --config projects/mono/holistic_path_prediction/holistic_path_prediction.py --device-ids ...
# Int stage
python3 tools/train.py --stage int_infer --config projects/mono/holistic_path_prediction/holistic_path_prediction.py --device-ids ...
```
## HPP推理
```
python3 tools/predict.py --stage xxx --config projects/mono/holistic_path_prediction/holistic_path_prediction.py --device-ids ...
```
## HPP编译
```
python3 tools/deploy/compile_standalone.py pt file path --input-size BxCxHxW --name xxx --opt xxx --march xxx
```