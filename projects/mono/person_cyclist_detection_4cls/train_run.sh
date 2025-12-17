# local
python tools/train.py --stage float --config projects/mono/person_cyclist_detection_4cls/multitask.py --pipeline-test --device-ids 1
python tools/train.py --stage qat --config projects/mono/person_cyclist_detection_4cls/multitask.py --pipeline-test --device-ids 0


# # aidi
python plugins/k8s_submit/submit.py --cluster share-3090-small-bcloud --config projects/mono/person_cyclist_detection_4cls/multitask.py