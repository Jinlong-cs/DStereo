#!/usr/bin/env bash

set -e

cd ../../

python3 plugins/code_stripping/code_stripping.py \
    --file-list plugins/release_package/example/codestrip_example.py \
    --src-dir ./ \
    --target-dir ./release

cd ./release

export PYTHONPATH=$(pwd):${PYTHONPATH}

cd docs/

make html