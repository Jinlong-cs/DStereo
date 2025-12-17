export PYTHONPATH=`pwd`
cd plugins/k8s_submit 
python3 submit.py \
--config ../../projects/bigmodel/configs/mvt4d/entry.py \
--cluster share-debug-queue-ucloud
