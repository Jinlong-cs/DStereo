#!/usr/bin/env bash

# get url
BUILD_URL=$1
echo "BUILD_URL=${BUILD_URL}"

# prepare hat
# set -e
export PYTHONPATH=$(pwd):${PYTHONPATH}

# rm -rf tmp_models
hdfs dfs -get hdfs://hobot-bigdata/user/rui.xu/horizon_algorithm_toolkit/release_models/bayes
mv bayes tmp_models
echo -e "\033[33mGet tmp_models from hdfs.\033[0m"

if [[ -e tmp_nuscenes ]];then
  echo -e "\033[33mTmp_nuscenes exist already.\033[0m"
else
  t1=`date +%s`
  mkdir ./tmp_nuscenes
  hdfs dfs -get hdfs://hobot-bigdata/user/zhenzhen.chao/data/nuscenes/database/nuscenes_infos_withdb.tar.gz ./tmp_nuscenes
  tar -xf ./tmp_nuscenes/nuscenes_infos_withdb.tar.gz -C ./tmp_nuscenes/
  t2=`date +%s`
  t=`echo $((${t2}-${t1}))`
  echo "Get tmp_nuscenes takes $t"
  echo -e "\033[33mGet tmp_nuscenes for centerpoint and lidar_multitask.\033[0m"
fi

# code stripping
rm -rf release
cd plugins/code_stripping/
python3 code_stripping.py --file-list configs/toolchain-file-list.py  --target-dir ../../release/HAT --override --toolchain-crop
cd ../../release/HAT

# prepare stripping hat
export PYTHONPATH=$(pwd):${PYTHONPATH}

# mount data, 
ln -s ../../tmp_data tmp_data
ln -s ../../tmp_orig_data tmp_orig_data
ln -s ../../tmp_pretrained_models tmp_pretrained_models
ln -s ../../tmp_models tmp_models
ln -s ../../tmp_nuscenes tmp_nuscenes
cp ../../projects/toolchain/dev/toolchain-config-list.py toolchain-config-list.py

res_file=./Result.txt
outlog=./result.log

if [[ -f ${res_file} ]]; then
    res_count=`wc -l ${res_file} |awk '{print $1}'`
else
    res_count=0
fi

# get base data
cp -r /horizon-bucket/HDLTAlgorithm/data/toolchain_hat/tmp_dump_data ./


function decide_by_shell(){
  if [ $? -eq 0 ];then
    res="PASS"
  else
    res="FAIL"
  fi
  model=$1
  stage=$2
  res_count=`expr ${res_count} + 1`
  echo "${res_count},${model},${stage},${res}!!" >> ${res_file}
}

function fly_fail_to_feishu(){
  webhook=${1}
  wtitle="TOOLCHAIN MODEL Daily TEST RESULT"
  outlogs=`cat ${2} | grep -i "fail"`
  outlog=${outlogs[*]}
  res=${outlog// /\\n}
  
  if [[ $res = "" ]];then
    res="All model pass!"
  fi
  echo $res
  wword="Jenkins URL: ${BUILD_URL}\nResult: ${res}"
  
  postdata="{\"msg_type\": \"post\", \
           \"content\": {\"post\": {\"zh_cn\":{\"title\": \"${wtitle}\", \
                                               \"content\": [[{\"tag\": \"text\", \"text\": \"${wword}\n\"}]]}}}}"
  echo ${postdata}
  curl ${webhook} -H 'Content-Type: application/json' -d "${postdata}"
}

skip_list="
configs/track_pred/motr_efficientnetb3_mot17_qim.py
"

config_list=`cat toolchain-config-list.py | grep -vE "\(|\)|#" | sed 's/\"//g' | sed 's/,//g'`

total_begin_t=`date +%s`
for cfg in ${config_list};
do
  if [[ $skip_list =~ ${cfg} ]];then
    continue
  fi
  begin_t=`date +%s`
  cp ${cfg} test.py
  net_name=`cat ${cfg} | grep "task_name = " | awk '{print $3}' | sed 's/\"//g'`
  echo '===============================================' >>$outlog 2>&1
  echo $net_name >>$outlog 2>&1

  #模型训练
  echo -e "\033[33mInfo=>start $net_name training.\033[0m"
  echo "Info=>start $net_name training." >>$outlog 2>&1
  sed -i 's/\(device_ids = \).*/\1[0,1]/g' test.py
  sed -i 's/\(.*batch_size_per_gpu = \).*/\11/g' test.py
  sed -i 's/stop_by="epoch",//g' test.py
  sed -i 's/num_steps=.*,/num_steps=1,/g' test.py
  sed -i 's/num_epochs=.*,/stop_by="step",num_steps=1,num_epochs=1,/g' test.py
  sed -i 's/.*val_callback,//g' test.py
  sed -i 's/ckpt_callback,//g' test.py
  sed -i 's/num_workers=.*,/num_workers=0,/g' test.py

  if [[ $net_name =~ 'torchvision' ]];then
    t1=`date +%s`
    python3 tools/train.py --config test.py --stage qat >>$outlog 2>&1
    decide_by_shell $net_name 'qat_train'
    t2=`date +%s`
		t=`echo $((${t2}-${t1}))`
		echo "$net_name,qat_train,$t," >>${res_file}
  elif [[ $net_name =~ 'detr3d_' ]] || [[ $net_name =~ 'petr_' ]] || [[ $net_name =~ 'bev_mt_ipm_temporal' ]];then
    echo "$net_name,train,dont,test" >>${res_file}
  else
    if [[ $net_name =~ '_cityscapes' ]] || [[ $net_name =~ '_imagenet' ]] ;then
      sed -i 's/\(.*batch_size_per_gpu = \).*/\12/g' test.py
    fi
    t1=`date +%s`
    python3 tools/train.py --config test.py --stage float >>$outlog 2>&1
    decide_by_shell $net_name 'float_train'
    t2=`date +%s`
    t=`echo $((${t2}-${t1}))`
    echo "$net_name,float_train,$t," >>${res_file}

    t1=`date +%s`
    python3 tools/train.py --config test.py --stage calibration >>$outlog 2>&1
    decide_by_shell $net_name 'calibration'
    t2=`date +%s`
    t=`echo $((${t2}-${t1}))`
    echo "$net_name,calibration,$t," >>${res_file}

    qat_limit=$(grep -r 'qat_trainer' test.py|wc -l)
    if [[ $qat_limit -ge 1 ]];then
      sed -i 's/\(qat_data\[\"batch_size\"\] = \).*/\11/g' test.py
      t1=`date +%s`
      python3 tools/train.py --config test.py --stage qat >>$outlog 2>&1
      decide_by_shell $net_name 'qat_train'
      t2=`date +%s`
      t=`echo $((${t2}-${t1}))`
      echo "$net_name,qat_train,$t," >>${res_file}
    fi
  fi

  #混合精度训练
  trainer=$(grep -w -n 'batch_processor = dict' test.py |tr -cd "[0-9]")
	mixattr=$(grep -w  'enable_amp' test.py |wc -l)
  change='\ \ \ \ \enable_amp=True,'
  
  if [[ $trainer -le 0 ]];then 
	  trainer=$(grep -w -n 'train_batch_processor = dict' test.py |tr -cd "[0-9]")
	fi   
	
	if [[ $mixattr -gt 0 ]];then 
    sed -i 's/enable_amp = False/enable_amp = True/g' test.py
  else
	  sed -i "${trainer}a  ${change} " test.py
  fi
    
	if [[ $qat_limit -ge 1 ]];then
    if [[ $net_name =~ 'detr3d_' ]] || [[ $net_name =~ 'petr_' ]] || [[ $net_name =~ 'bev_mt_ipm_temporal' ]];then
      echo "$net_name,mix,train,dont,test" >>${res_file}
    fi
    t1=`date +%s`
    python3 tools/train.py --config test.py --stage qat >>$outlog 2>&1
    decide_by_shell $net_name 'mix'
	  t2=`date +%s`
	  t=`echo $((${t2}-${t1}))`
	  echo "$net_name,mix,$t," >>${res_file}
	fi	

  #网络predict
  echo -e "\033[33mInfo=>start $net_name predict.\033[0m"
  echo "Info=>start $net_name predict." >>$outlog 2>&1
  if [[ $net_name =~ 'torchvision' ]];then
    int_infer_predictor=$(grep -w -n 'int_infer_predictor = dict' test.py | tr -cd "[0-9]")
    int_infer_predictor=$((${int_infer_predictor}+1))
    change='\ \ \ \ \stop_by="step",num_steps=1,'
    sed -i "${int_infer_predictor}a  ${change} " test.py
    t1=`date +%s`
    python3 tools/predict.py --config test.py --stage int_infer >>$outlog 2>&1
    decide_by_shell $net_name 'int_infer_predict'
    t2=`date +%s`
    t=`echo $((${t2}-${t1}))`
    echo "$net_name,int_infer_predict,$t," >>${res_file}
  else
    if [[ $net_name =~ '_nuscenes' ]] || [[ $net_name =~ 'bev_' ]];then
      sed -i 's/val_metric_updater,//g' test.py
      sed -i 's/metric_updater,//g' test.py
    fi
    float_predictor=$(grep -w -n 'float_predictor = dict' test.py | tr -cd "[0-9]")
    float_predictor=$((${float_predictor}+1))
    change='\ \ \ \ \stop_by="step",num_steps=1,'
    if [[ $net_name =~ 'yolo_mobilenetv1_' ]];then
      change='\ \ \ \ \stop_by="step",num_steps=100,'
    fi
    sed -i "${float_predictor}a  ${change} " test.py
    t1=`date +%s`
    python3 tools/predict.py --config test.py --stage float >>$outlog 2>&1
    decide_by_shell $net_name 'float_predict'
    t2=`date +%s`
    t=`echo $((${t2}-${t1}))`
    echo "$net_name,float_predict,$t," >>${res_file}

    if [[ $net_name =~ 'pwcnet_pwcnetneck_flyingchairs' ]];then
      sed -i 's/\(.*batch_size_per_gpu = \).*/\12/g' test.py
    fi
    int_infer_predictor=$(grep -w -n 'int_infer_predictor = dict' test.py | tr -cd "[0-9]")
    int_infer_predictor=$((${int_infer_predictor}+1))
    sed -i "${int_infer_predictor}a  ${change} " test.py
    t1=`date +%s`
    python3 tools/predict.py --config test.py --stage int_infer >>$outlog 2>&1
    decide_by_shell $net_name 'int_infer_predict'
    t2=`date +%s`
    t=`echo $((${t2}-${t1}))`
    echo "$net_name,int_infer_predict,$t," >>${res_file}
  fi

  #网络align_bpu_validation
  echo -e "\033[33mInfo=>start $net_name align_bpu_validation.\033[0m"
  echo "Info=>start $net_name align_bpu_validation." >>$outlog 2>&1
  align_bpu_predictor=$(grep -w -n 'align_bpu_predictor = dict' test.py | tr -cd "[0-9]")
  if [[ $align_bpu_predictor -le 0 ]];then 
    align_bpu_predictor=$(grep -n 'align_bpu_predictor = ' test.py | tr -cd "[0-9]")
    change='align_bpu_predictor["stop_by"]="step"'
    sed -i "${align_bpu_predictor}a  ${change} " test.py
    change='align_bpu_predictor["num_steps"]=1'
    sed -i "${align_bpu_predictor}a  ${change} " test.py
  else
    align_bpu_predictor=$((${align_bpu_predictor}+1))
    change='\ \ \ \ \stop_by="step",num_steps=1,'
    sed -i "${align_bpu_predictor}a  ${change} " test.py
  fi

  if [[ $net_name =~ 'detr3d_' ]] || [[ $net_name =~ 'petr_' ]] || [[ $net_name =~ 'bev_' ]];then
    echo "$net_name,align_bpu_validation,dont,test" >>${res_file}
  else
    t1=`date +%s`
    python3 tools/align_bpu_validation.py --config test.py >>$outlog 2>&1
    decide_by_shell $net_name 'validate'
    t2=`date +%s`
    t=`echo $((${t2}-${t1}))`
    echo "$net_name,align_bpu_validation,$t," >>${res_file}
  fi

  #网络infer
  echo -e "\033[33mInfo=>start $net_name infer.\033[0m"
  echo "Info=>start $net_name infer." >>$outlog 2>&1
  t1=`date +%s`
  python3 tools/infer.py -c test.py >>$outlog 2>&1
  decide_by_shell $net_name 'infer'
  t2=`date +%s`
	t=`echo $((${t2}-${t1}))`
	echo "$net_name,infer,$t," >>${res_file}
  
  #网络计算量
  echo -e "\033[33mInfo=>start $net_name calops.\033[0m"
  echo "Info=>start $net_name calops." >>$outlog 2>&1
	t1=`date +%s`
  if [[ $net_name =~ 'petr_' ]];then
    python3 tools/calops.py --config test.py --method hook >calops.log 2>&1
    decide_by_shell $net_name 'calops'
  else
    python3 tools/calops.py --config test.py >calops.log 2>&1
    decide_by_shell $net_name 'calops'
  fi
	t2=`date +%s`
	t=`echo $((${t2}-${t1}))`
	echo "$net_name,calops,$t," >>${res_file}	
  cat calops.log >>$outlog 2>&1 
  result_calops_search_1=$(cat calops.log|grep -Eo 'FLOPs: [0-9\.]*')#保留两位小数
  result_calops_search=$(echo ${result_calops_search_1#*: })
  result_calops=$(echo $result_calops_search |awk '{printf "%.2f", $1}')
  echo  "$net_name,${result_calops},," >> calops_res.log 2>&1
  
  # numeric consistent test
  echo -e "\033[33mInfo=>start $net_name numeric test.\033[0m"
  t1=`date +%s`
  if [[ $net_name =~ 'torchvision' ]];then
    echo "$net_name,float_trainer,not,exist" >>${res_file}
  else
    echo "Info=>start $net_name training numeric test." >>$outlog 2>&1
    python3 tests/test_numeric_consistent.py --config ${cfg} --action train >>$outlog 2>&1
    decide_by_shell $net_name 'train_numeric_consistent'

    echo "Info=>start $net_name prediction numeric test." >>$outlog 2>&1
    python3 tests/test_numeric_consistent.py --config ${cfg} --action predict >>$outlog 2>&1
    decide_by_shell $net_name 'predict_numeric_consistent'
  fi
  t2=`date +%s`
  t=`echo $((${t2}-${t1}))`
	echo "$net_name,numeric_consistent,$t," >>${res_file}

  # export onnx test
  echo -e "\033[33mInfo=>start $net_name export onnx.\033[0m"
  echo "Info=>start $net_name export onnx." >>$outlog 2>&1

  t1=`date +%s`
  if [[ $net_name =~ 'horizon_swin' ]];then
    echo "$net_name,export_onnx,not,test" >>${res_file}
  else
    python3 tools/export_onnx.py --config test.py >>$outlog 2>&1
    decide_by_shell $net_name 'export_onnx'
  fi
  t2=`date +%s`
	t=`echo $((${t2}-${t1}))`
	echo "$net_name,export_onnx,$t," >>${res_file}	
  
  end_t=`date +%s`
  t=`echo $((${end_t}-${begin_t}))`
  echo "$net_name,pipeline,$t," >>${res_file}

  rm -rf test.py
done

total_end_t=`date +%s`
t=`echo $((${total_end_t}-${total_begin_t}))`
echo "all,model,takes,$t," >>${res_file}
logs=`cat $res_file`
for i in $logs;do
  echo $i
done

# save logs
cp $outlog /horizon-bucket/HDLTAlgorithm/data/toolchain_hat/
cp $res_file /horizon-bucket/HDLTAlgorithm/data/toolchain_hat/
cp calops_res.log /horizon-bucket/HDLTAlgorithm/data/toolchain_hat/

# send to feishu
webhook="https://open.feishu.cn/open-apis/bot/v2/hook/c4a8747f-2cf4-44d3-8138-8df7686c4b24"
fly_fail_to_feishu $webhook $res_file
