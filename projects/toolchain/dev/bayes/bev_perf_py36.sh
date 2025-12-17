#!/usr/bin/env bash

set -e
export PYTHONPATH=`pwd`:${PYTHONPATH}

make basic-data
rm -rf bev_release_models*
rm -rf tmp_models*

hdfs dfs -get hdfs://hobot-bigdata/user/kongtao.hu/horizon_algorithm_toolkit/python3.6/bev_release_models.tgz
tar -xvf bev_release_models.tgz
mv bev_release_models tmp_models

find ./ -name "compile" | xargs rm -rf

config_list="
configs/bev/bev_mt_ipm.py
configs/bev/bev_mt_lss.py
configs/bev/bev_mt_gkt.py
configs/bev/bev_mt_ipm_temporal.py
configs/bev/detr3d_efficientnetb3_nuscenes.py
configs/bev/petr_efficientnetb3_nuscenes.py
"

for cfg in ${config_list};
do
  python3 tools/compile_perf.py --config ${cfg}
done

cd tmp_models
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

# update
cp projects/toolchain/dev/bayes/BEV_Result_py38.LOG tmp_models/Result.LOG
mv tmp_models bev_release_models
rm -rf bev_release_models.tgz
tar -zcvf bev_release_models.tgz bev_release_models/
hdfs dfs -rm hdfs://hobot-bigdata/user/kongtao.hu/horizon_algorithm_toolkit/python3.6/bev_release_models.tgz
hdfs dfs -copyFromLocal bev_release_models.tgz hdfs://hobot-bigdata/user/kongtao.hu/horizon_algorithm_toolkit/python3.6/

# rm -rf bev_release_models*
