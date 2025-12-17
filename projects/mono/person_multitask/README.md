一、如何执行
默认进入HAT根目录；
1. 训练
python tools/train.py -c projects/mono/person_multitask/entry.py -s float

2. 提交集群
python plugins/k8s_submit/submit.py -c projects/mono/person_multitask/entry.py --cluster xxx

3. 预测
cd projects/mono/person_multitask/infer; python infer_job_runner.py

4. 评测
python projects/mono/person_multitask/tools/xxx_eval.py （根据任务类型选择不同脚本）

5. 编译
sh projects/mono/person_multitask/tools/compile_pt.sh /path/int_infer-deploy-checkpoint-last.pt