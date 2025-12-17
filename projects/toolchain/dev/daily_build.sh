#!/usr/bin/env bash

set -e

export PATH=~/.local/bin:/root/.local/bin:/home/cicd/.local/bin/:$PATH

BUILD_URL=$1
# add model fast pipeline tests
# install deps
echo "---------------------install toolchain reqs------------------------"
pip_ext="-i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc"
pip3 install --user numpy==1.19.5 ${pip_ext}
pip3 install --use-deprecated=legacy-resolver --user -r projects/toolchain/requirements/build.txt ${pip_ext}
echo "------------------install success!------------------"
echo "------------------pip list------------------"
pip3 list
pwd & ls
sh projects/toolchain/dev/model_fast_pipeline.sh ${BUILD_URL}

# Need Refactor
# # install deps
# pip_ext="-i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc"
# pip3 install --use-deprecated=legacy-resolver --user -r requirements.txt ${pip_ext}
# pip3 install -U hbdk -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
# pip3 install --user -U horizon-plugin-pytorch -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/cu102/torch1102 --trusted-host art-internal.hobot.cc ${pip_ext}

# # prepare hat
# export PYTHONPATH=$(pwd):${PYTHONPATH}

# # code stripping
# rm -rf release
# cd plugins/code_stripping/
# python3 code_stripping.py --file-list configs/toolchain-file-list.py --target-dir ../../release/HAT --overrid
# cd ../../release/HAT

# # prepare stripping hat
# export PYTHONPATH=$(pwd):${PYTHONPATH}

# # mount data, 
# ln -s ../../tmp_data tmp_data
# ln -s ../../tmp_orig_data tmp_orig_data
# ln -s ../../tmp_models tmp_orig_models

# # merge project/toolchain/, mainly docs
# cp -r projects/toolchain/docs/examples/ docs/source/
# sed -i 's/-W//g' docs/Makefile
# line=$(grep -w -n 'model_zoo/model_zoo.md' docs/source/index.rst |tr -cd "[0-9]")
# sedline=$(expr ${line} + 1)
# IFS=''
# cat projects/toolchain/docs/sub_index.rst| while read line
# do
#   sed -i "${sedline}a\ $line" docs/source/index.rst
#   sedline=$(expr ${sedline} + 1)
# done
# sed -i "s/\ \.. toctree::/.. toctree::/g" docs/source/index.rst

# # merge project/toolchain/tools
# cp -r projects/toolchain/tools/* tools/

# # run tests
# echo "---------------------run test------------------------"
# make serial-unit-test
# make unit-test
# make intergration-tests
# echo "------------------run test success!------------------"

# # run doc tests
# echo "-------------------run doc test----------------------"
# cd docs && make html && cd ..
# echo "-----------------doc test success!-------------------"

# # build release packages
# echo "---------------- add release packages----------------"
# ./projects/toolchain/dev/release_packages.sh
# echo "-------------add release packages success!-----------"

# # build release modelzoo
# echo "---------------- add release modelzoo----------------"
# ./projects/toolchain/dev/model_metric/run.sh
# echo "-------------add release modelzoo success!-----------"

# # TODO (zhigang.yang, ?), upload gallery for release
# echo "----------------upload release package---------------"
# ls release_package.tgz
# ls model_zoo.tgz
# echo "-----------upload release package success!-----------"
