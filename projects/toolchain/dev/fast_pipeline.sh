#!/usr/bin/env bash
# experimental

set -e

ln -s tmp_orig_models/bayes_release_models/ tmp_pretrained_models
# prevent write operations on gpfs
rm tmp_models

config_list="
configs/segmentation/unet.py
"

for cfg in ${config_list};
do
  cp ${cfg} test.py
  sed -i 's/\(device_ids = \).*/\1[0,1]/g' test.py
  sed -i 's/num_epochs=.*,/num_epochs=1,/g' test.py
  python3 tools/train.py --config test.py --step float
  python3 tools/train.py --config test.py --step qat
  python3 tools/train.py --config test.py --step int_infer
  python3 tools/compile_perf.py --config test.py
  rm -rf test.py
done
