#!/usr/bin/env bash

set -e

OUTPUT_DIR="release_package"
if [ -d "$OUTPUT_DIR" ]; then
    rm -r $OUTPUT_DIR
fi

# code 
cd ../..
python3 plugins/code_stripping/code_stripping.py \
    --file-list projects/pilot/dev/file_list/internal_test_file_list.py \
    --src-dir ./ \
    --target-dir projects/pilot/$OUTPUT_DIR \
    --clip-code
cd projects/pilot
export PYTHONPATH="$(pwd)/$OUTPUT_DIR":${PYTHONPATH}

cd $OUTPUT_DIR/docs/

make html
