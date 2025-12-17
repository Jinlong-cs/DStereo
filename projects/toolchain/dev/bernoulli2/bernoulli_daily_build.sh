#!/usr/bin/env bash

set -e

export PATH=/root/.local/bin:/home/cicd/.local/bin/:$PATH

# install deps
# pip_ext="-i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc"
# pip3 install --use-deprecated=legacy-resolver --user -r requirements.txt ${pip_ext}
# pip3 install -U --user horizon_plugin_pytorch_cu102 ${pip_ext}

# code stripping
cd plugins/code_stripping/
python3 code_stripping.py --file-list plugins/code_stripping/configs/toolchain-file-list-bernoulli2.py --target-dir ../../release/HAT --override --toolchain-crop
cd ../../release/HAT
# prepare stripping hat
export PYTHONPATH=`pwd`:${PYTHONPATH}

# mount data, 
# ln -s ../../tmp_data tmp_data
# ln -s ../../tmp_orig_data tmp_orig_data
# ln -s ../../tmp_models tmp_orig_models

# merge project/toolchain/tools
#cp -r projects/toolchain/tools/* tools/
#cp -r projects/toolchain/configs/* configs/

# run tests
echo "-----------------doc test success!-------------------"

# build release packages
echo "---------------- add release packages----------------"
sh projects/toolchain/dev/bernoulli_release_packages.sh
echo "-------------add release packages success!-----------"

# build release modelzoo
echo "---------------- add release modelzoo----------------"
if [ $SKIP_MODEL_METRIC != "1" ]
then
    ./projects/toolchain/dev/model_metric/run.sh
else
    hdfs dfs -get hdfs://hobot-bigdata/user/kongtao.hu/horizon_algorithm_toolkit/bernoulli/python3.6/release_hbms.tgz
    hdfs dfs -get hdfs://hobot-bigdata/user/kongtao.hu/horizon_algorithm_toolkit/bernoulli/python3.6/release_models.tgz
    mv release_hbms.tgz release_hbms-$RELEASE_VERSION.tgz
    mv release_models.tgz release_models-$RELEASE_VERSION.tgz
fi
echo "-------------add release modelzoo success!-----------"

# TODO (kongtao.hu, ?), upload gallery for release
echo "----------------upload release package---------------"
ls release_package*.tgz
ls release_models*.tgz
hdfs dfs -copyFromLocal release_models*.tgz hdfs://hobot-bigdata/user/kongtao.hu/horizon_algorithm_toolkit/bernoulli/v$RELEASE_VERSION/py36/
hdfs dfs -copyFromLocal release_hbms* hdfs://hobot-bigdata/user/kongtao.hu/horizon_algorithm_toolkit/bernoulli/v$RELEASE_VERSION/py36/
hdfs dfs -copyFromLocal release_package*.tgz hdfs://hobot-bigdata/user/kongtao.hu/horizon_algorithm_toolkit/bernoulli/v$RELEASE_VERSION/py36/
echo "-----------upload release package success!-----------"
