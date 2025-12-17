export PYTHONPATH=`pwd`
cfg=../../$1/entry.py

cd plugins/k8s_submit

if [ ${debug} ]
then
    # debug cluster
    default_cluster=share-debug-bcloud 
elif [ ${aidi_eval} ]
then
    # eval cluster
    default_cluster=share-3090-small-bcloud
else
    # train cluster
    default_cluster=share-3090-small-bcloud
fi

cluster=${default_cluster}

echo $cfg
echo $cluster
python3 submit.py --config ${cfg} --cluster ${cluster}
