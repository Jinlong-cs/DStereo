#!/usr/bin/env bash

bucket_root=`pwd`/tmp_bucket
if [ ! -d ${bucket_root} ];then
    mkdir -p ${bucket_root}
fi

hitc mount show

read -p "bucket name (split by blank)? " answer
if [[ $answer = "" ]]; then
    echo "Please input bucket name"
    exit
else
    for bucket in $(echo $answer | tr " " "\n"); do
        hitc mount exec $bucket ${bucket_root}/${bucket}
        echo "mount ${bucket} to ${bucket_root}"
    done
fi
