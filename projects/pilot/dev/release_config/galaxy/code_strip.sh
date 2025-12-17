#!/usr/bin/env bash

set -e

cd ../../

python3 plugins/code_stripping/code_stripping.py \
    --file-list projects/pilot/dev/file_list/galaxy_file_list.py \
    --src-dir ./ \
    --target-dir ./release \
    --clip-code  \
cd ./release

export PYTHONPATH=$(pwd):${PYTHONPATH}

cd docs/

make html