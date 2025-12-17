#!/usr/bin/env bash

set -e
export PYTHONPATH=`pwd`:${PYTHONPATH}

#rm -rf release_models*
#rm -rf release_hbms*
#rm -rf tmp_models*

hdfs dfs -get hdfs://hobot-bigdata/user/rui.xu/horizon_algorithm_toolkit/release_models/bayes
mv bayes tmp_models

find ./ -name "compile" | xargs rm -rf

cp projects/toolchain/dev/toolchain-config-list.py toolchain-config-list.py

config_list=`cat toolchain-config-list.py | grep -vE "\(|\)|#" | sed 's/\"//g' | sed 's/,//g'`

for cfg in ${config_list};
do
  echo ${cfg}
  python3 tools/deploy/compile_perf.py --config  projects/toolchain/${cfg} --jobs 32
done

cd tmp_models
rm -rf *.LOG
find ./ -name "*.html" | xargs rm -rf
find ./ -name "*.json" | xargs rm -rf
find ./ -name "*.hbir" | xargs rm -rf

hbm=$(find ./ -name *.hbm)
for i in $hbm; do
    dir=${i%/*}
    cd $dir
    mv *hbm model.hbm
    cd -
done
cd ..

cp -r tmp_models release_hbms
find ./release_hbms -name "*.pth.tar" | xargs rm -rf
find ./release_hbms -name "*.pth" | xargs rm -rf
find ./release_hbms -name "*.pt" | xargs rm -rf

# update
cp projects/toolchain/dev/Result.LOG  tmp_models/Result.LOG
cp -r tmp_models release_models
find ./release_models -name "*.hbm" | xargs rm -rf

hdfs dfs -rm -r   hdfs://hobot-bigdata/user/rui.xu/horizon_algorithm_toolkit/bayes/python3.8/*
hdfs dfs -copyFromLocal release_models hdfs://hobot-bigdata/user/rui.xu/horizon_algorithm_toolkit/bayes/python3.8/
hdfs dfs -copyFromLocal release_hbms   hdfs://hobot-bigdata/user/rui.xu/horizon_algorithm_toolkit/bayes/python3.8/
