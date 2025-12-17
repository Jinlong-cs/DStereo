#!/usr/bin/env bash

# set -e
export PYTHONPATH=`pwd`:${PYTHONPATH}

# rm -rf release_models*
# rm -rf release_hbms*
# rm -rf tmp_models*

# hdfs dfs -get hdfs://hobot-bigdata/user/rui.xu/horizon_algorithm_toolkit/release_models/bayes
# mv bayes tmp_models

mkdir -p perf_logs
rm -rf perf_logs/*

find ./ -name "compile" | xargs rm -rf
cp projects/toolchain/dev/nash/toolchain-config-list.py toolchain-config-list.py

config_list=`cat toolchain-config-list.py | grep -vE "\(|\)|#" | sed 's/\"//g' | sed 's/,//g'`

for cfg in ${config_list};
do
  cfg_path="projects/toolchain/${cfg}"
  task_name=`cat ${cfg_path} | grep "task_name = " | awk '{print $3}' | sed 's/\"//g'`
  # resultlogfile="./perf_logs/${task_name}_export_hbir.log"
  resultlogfile_compile="./perf_logs/${task_name}_compile.log"
  # echo "${task_name} export start!!!"
  # python3 tools/deploy/export_hbir.py --config ${cfg_path} >> $resultlogfile 2>&1
  # echo "${task_name} export stop!!!"
  # echo "start compile!!"
  echo "${task_name} compile start!!!"
  python3 projects/toolchain/tools/compile_perf_hbir.py --config ${cfg_path} --jobs 32 >> $resultlogfile_compile 2>&1
  echo "${task_name} compile stop!!!"
done

# cd tmp_models
# rm -rf *.LOG
# find ./ -name "*.html" | xargs rm -rf
# find ./ -name "*.json" | xargs rm -rf
# find ./ -name "*.hbir" | xargs rm -rf

# hbm=$(find ./ -name *.hbm)
# for i in $hbm; do
#     dir=${i%/*}
#     cd $dir
#     mv *hbm model.hbm
#     cd -
# done
# cd ..

# cp -r tmp_models release_hbms
# find ./release_hbms -name "*.pth.tar" | xargs rm -rf
# find ./release_hbms -name "*.pth" | xargs rm -rf
# find ./release_hbms -name "*.pt" | xargs rm -rf

# # # update
# cp projects/toolchain/dev/Result.LOG  tmp_models/Result.LOG
# mv tmp_models release_models
# find ./release_models -name "*.hbm" | xargs rm -rf

# hdfs dfs -rm -r   hdfs://hobot-bigdata/user/rui.xu/horizon_algorithm_toolkit/bayes/python3.8/hbdk4/*
# hdfs dfs -copyFromLocal release_models hdfs://hobot-bigdata/user/rui.xu/horizon_algorithm_toolkit/bayes/python3.8/hbdk4/
# hdfs dfs -copyFromLocal release_hbms   hdfs://hobot-bigdata/user/rui.xu/horizon_algorithm_toolkit/bayes/python3.8/hbdk4/
