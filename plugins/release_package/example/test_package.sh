#!/usr/bin/env bash
set -e

cd release
export PYTHONPATH=`pwd`:${PYTHONPATH}
python3 tools/prepare_bucket.py --bucket "HDLTAlgorithm" --mount --create-link
pytest -s tests/unit_tests/data/datasets/test_imagenet.py
