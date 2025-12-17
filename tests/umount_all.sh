#!/usr/bin/env bash

bucket_root=`pwd`/tmp_bucket
if [ ! -d ${bucket_root} ];then
    exit    
fi

for bucket in `ls ${bucket_root}`; do
    hitc mount umount ${bucket_root}/${bucket}
done
