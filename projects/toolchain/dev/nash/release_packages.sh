#!/usr/bin/env bash

set -e

export PYTHONPATH=`pwd`:$PYTHONPATH

export RELEASE_VERSION=3.0.1
export OE_VERSION=2.5.2
export HORIZON_PLUGIN_PYTORCH_VERSION=1.6.3
export HEMAT_VERSION=1.0.4
export OLD_RELEASE_VERSION=2.0.10
export OLD_OE_VERSION=1.1.44
export SKIP_TEST=1
export SKIP_MODEL_METRIC=1

hdfs dfs -mkdir -p hdfs://hobot-bigdata/user/rui.xu/horizon_algorithm_toolkit/bayes/v$RELEASE_VERSION/py38/

# package  release
cd plugins/code_stripping/
python3 code_stripping.py --file-list ../../projects/toolchain/dev/nash/toolchain-file-list.py --target-dir ../../release/HAT --override --toolchain-crop
cd ../../release/HAT

export PYTHONPATH=`pwd`:${PYTHONPATH}

pip_ext="-i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc"

mkdir -p release_package/packages
mkdir -p release_package/scripts

# update version
# sed -i s/"__version__ = \"/__version__ = \"$RELEASE_VERSION\" #"/g hat/version.py
a=$(grep -ri "__version__" hat/version.py | grep  -w -Eo '[0-9\.]*')
sed -i "s/$a/$RELEASE_VERSION/g" hat/version.py

sed -i "s/RELEASE_VERSION/$RELEASE_VERSION/g" `grep -rl "RELEASE_VERSION" --include="*.md" ./configs`

pip3 download --no-deps hbdk4-compiler==4.0.5 -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
pip3 download --no-deps horizon-plugin-pytorch==1.10.2 -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/cu116/torch1130/ -i http://pypi.hobot.cc/simple --extra-index-url http://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc --trusted-host art-internal.hobot.cc

mv hbdk*whl release_package/packages/
mv horizon_plugin_pytorch*whl release_package/packages/

pip3 uninstall -y hbdk hbdk-internal horizon_plugin_pytorch_cu111
pip3 install release_package/packages/*whl ${pip_ext}
python3 setup.py sdist bdist_wheel

cp -r dist/horizon_torch_samples-*whl release_package/packages

rm -rf dist

# prepare scripts
mkdir -p release_package/scripts/configs/
mkdir -p release_package/scripts/tools/
mkdir -p release_package/scripts/examples/

cp -r configs/*  release_package/scripts/configs/
cp -r tools/* release_package/scripts/tools/
cp -r examples/* release_package/scripts/examples/

# prepare tgz
find ./release_package/ | grep -E "__pycache__" | xargs rm -rf
tar -zcvf release_package-$RELEASE_VERSION.tgz release_package

# pack release models
hdfs dfs -get hdfs://hobot-bigdata/user/rui.xu/horizon_algorithm_toolkit/bayes/python3.8/hbdk4/release_hbms
hdfs dfs -get hdfs://hobot-bigdata/user/rui.xu/horizon_algorithm_toolkit/bayes/python3.8/hbdk4/release_models
cp ../../projects/toolchain/dev/Result.LOG  release_models/Result.LOG
tar -zcvf release_hbms-$RELEASE_VERSION.tgz release_hbms
tar -zcvf release_models-$RELEASE_VERSION.tgz release_models

ls release_package*.tgz
ls release_models*.tgz
hdfs dfs -copyFromLocal release_models*.tgz hdfs://hobot-bigdata/user/rui.xu/horizon_algorithm_toolkit/bayes/v$RELEASE_VERSION/py38/
hdfs dfs -copyFromLocal release_hbms*.tgz hdfs://hobot-bigdata/user/rui.xu/horizon_algorithm_toolkit/bayes/v$RELEASE_VERSION/py38/
hdfs dfs -copyFromLocal release_package*.tgz hdfs://hobot-bigdata/user/rui.xu/horizon_algorithm_toolkit/bayes/v$RELEASE_VERSION/py38/

hdfs dfs -ls hdfs://hobot-bigdata/user/rui.xu/horizon_algorithm_toolkit/bayes/v$RELEASE_VERSION/py38/

# upload models
wget -q http://file.ddk.hobot.cc/oe_file/lftp_compiled.tar.gz
tar zxf lftp_compiled.tar.gz
rm -rf lftp_compiled.tar.gz
./lftp/bin/lftp -e "mirror -R ./release_models/ /openexplorer/horizon_torch_samples/$RELEASE_VERSION/py38/modelzoo/qat_origin_modelzoo/;exit" -u 'oe_admin,L%CE8Dbw' vrftp.horizon.ai
rm -rf lftp
