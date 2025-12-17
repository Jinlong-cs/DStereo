#!/bin/bash

interval=${INTERVAL}
p_gateway="${PUSHGATEWAY}"
podname="${PODNAME}"
ns=${NAMESPACE}

if [ -z $interval ];then
	interval=1
fi

if [ -z $ns ];then
        ns="scm"
fi

if [ -z $p_gateway ];then
        p_gateway="http://prometheus2-pushgateway.monitoring.svc:9091"
fi



while true;
do
	nvidia-smi --format=csv,noheader --query-gpu=index,fan.speed,utilization.gpu,memory.used,memory.total > rec

	while read line;
	do
		gpuid=`echo "$line"|awk -F '[ , %]+' '{print $1}'`
		gpu_Util=`echo "$line"|awk -F '[ , %]+' '{print $3}'`
		gpu_memory_used=`echo "$line"|awk -F '[ , %]+' '{print $4}'`
		gpu_memory_total=`echo "$line"|awk -F '[ , %]+' '{print $6}'`
		echo "gpu_util $gpu_Util" |curl --data-binary @- $p_gateway/metrics/job/scm_"$podname"_gpu_"$gpuid"/namespace/"$ns"
		echo "gpu_memory_used  $gpu_memory_used" | curl --data-binary @- $p_gateway/metrics/job/scm_"$podname"_gpu_"$gpuid"/namespace/"$ns"
		echo "gpu_memory_total $gpu_memory_total" | curl --data-binary @- $p_gateway/metrics/job/scm_"$podname"_gpu_"$gpuid"/namespace/"$ns"
	done < rec
	sleep $interval
done
