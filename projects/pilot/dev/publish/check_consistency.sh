#!/bin/bash

JOB=https://haihui.chen:1111a9c180388f1f1d18a976424497a5d0@ci.hobot.cc/view/QA/job/QA-Auto/job/QA02_DEV/job/$1
PROJECT=$2
MODEL_TYPE=$3
MODEL_VERSION=$4
MODEL_URL=$5
ALG_RESULT_URL=$6

# 远程触发CI build
prev_number=0
cur_number=0
while [ $((cur_number - prev_number)) -ne 1 ];
do
  prev_number=`curl --silent ${JOB}/lastBuild/buildNumber`
  #echo "上一次BuildNumber：${prev_number}"
  # 触发新Jenkins Build
  curl -X POST "${JOB}/buildWithParameters" \
  -d "PROJECT=${PROJECT}"\
  -d "MODEL_TYPE=${MODEL_TYPE}"\
  -d "MODEL_VERSION=${MODEL_VERSION}"\
  -d "MODEL_URL=${MODEL_URL}"\
  -d "ALG_RESULT_URL=${ALG_RESULT_URL}"

  sleep 20
  cur_number=`curl --silent ${JOB}/lastBuild/buildNumber`
  #echo "当前BuildNumber：${cur_number}"
done

# Jenkins Build进行中，等待16min
sleep 960

# 确认Build状态，直至其为执行完毕状态
build_result="None"
while [ "${build_result}" = "None" ];
do
  # 1min后，尝试获取Build结果
  sleep 60
  json_res=`curl --silent ${JOB}/${cur_number}/api/json?pretty=true`
  build_result=`echo $json_res | python -c 'import sys, json; print(json.load(sys.stdin)["result"])'`
  #echo ${build_result}
done

# 获取返回值
artifacts_len="0"
return_value="invalid"
if [ "${build_result}" = "SUCCESS" ]; then
  artifacts_len=`echo $json_res | python -c 'import sys, json; print(len(json.load(sys.stdin)["artifacts"]))'`
  if [ "${artifacts_len}" != "0" ]; then
    # Build执行成功，获取结果
    str=`echo $json_res | python -c 'import sys, json; print(json.load(sys.stdin)["artifacts"][0]["fileName"])'`
    return_value=${str:14:5}
  fi
fi
# result_value有true，false，invalid三种值；
# 当Build执行失败，强制退出，对比条件未满足时，result_value值为invalid
# 正常进入对比功能后，结果一致时，result_value值为true，结果不一致时为false
echo "${cur_number}:${return_value}"
