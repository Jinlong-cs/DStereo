#!/usr/bin/env bash

set -e

test_level=${1}

OUTPUT_DIR="release_package"
if [ -d "$OUTPUT_DIR" ]; then
    rm -r $OUTPUT_DIR
fi

PROJECT_BASE=projects/superparking
cd ../..
python3 plugins/code_stripping/code_stripping.py \
    --file-list ${PROJECT_BASE}/dev/file_list/internal_full_file_list.py \
    --src-dir ./ \
    --target-dir ${PROJECT_BASE}/$OUTPUT_DIR \
    --clip-code
cd ${PROJECT_BASE}
export PYTHONPATH="$(pwd)/$OUTPUT_DIR":${PYTHONPATH}

if [[ ${test_level} != "local" ]]
then
cd $OUTPUT_DIR/docs/ # to be completed
make html
fi
