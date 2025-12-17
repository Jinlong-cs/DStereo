#!/usr/bin/env bash

set -e
export PATH=~/.local/bin:~/bin/$PATH
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "------------------- start code_stripping... ----------------------"
python3 projects/halo/cv/tools/release/release.py
python3 plugins/code_stripping/code_stripping.py --file-list halocv_release.py --src-dir ./
echo "------------------- end code_stripping... ----------------------"

cd release
export PYTHONPATH=$(pwd):$PYTHONPATH  # reset PYTHONPATH=xxx/release:$PYTHONPATH

# TODO: add halo project test code
# add dependencies
pip3 install click==8.0.2 ${pip_ext}
# face3d
pip3 install --user kornia lpips==0.1.4 ${pip_ext}
# hand3d  chumpy version 0.70
pip3 install --user smplx[all] chumpy ${pip_ext}
# chumpy can be upgraded to 0.71, which works with numpy >= 1.24.0
# pip install git+https://github.com/mattloper/chumpy
# uninstall memray for cicd
pip3 uninstall memray -y ${pip_ext}
pip3 list

# run unit test
ln -s /horizon-bucket/HDLTAlgorithm/data/pack_data tmp_data
ln -s /horizon-bucket/HDLTAlgorithm/data/orig_data tmp_orig_data
ln -s /horizon-bucket/HDLTAlgorithm/models/bayes_release_models tmp_pretrained_models
pytest tests/unit_tests -s -x -n 8
