import os

cmd = "python3 tools/predict.py --config projects/cloudmodel/configs/cloudsparse4d/model_configs/v1.1/entry.py --stage float --device-ids 0"
os.system(cmd)
