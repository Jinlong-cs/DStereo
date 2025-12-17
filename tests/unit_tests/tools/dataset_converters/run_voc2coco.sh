#!/usr/bin/env bash

set -e

python3 tools/data/voc2coco.py \
-p tmp_data/voc/trainval_lmdb \
-o trainval_pascal_voc.json

python3 tools/data/voc2coco.py \
-p tmp_data/voc/test_lmdb \
-o test_pascal_voc.json

