export HAT_BUCKET="/horizon-bucket"
export config="projects/planning/configs/value_iter_net.py"
export hat_dir="/home/users/zhiqiang03.zhang/0_code/HAT"
export PYTHPATH=${PATHPATH}:${PATHPATH}

cd ${hat_dir}
python3 -W ignore ${hat_dir}/tools/train.py --config ${config} --stage float --level 40
python3 -W ignore ${hat_dir}/tools/train.py --config ${config} --stage qat --level 40
python3 -W ignore ${hat_dir}/tools/train.py --config ${config} --stage int_infer --level 40

# 模型验证，并不是每次都需要
# python3 -W ignore ${hat_dir}/tools/predict.py --config ${config} --stage float
# python3 -W ignore ${hat_dir}/tools/predict.py --config ${config} --stage qat
# python3 -W ignore ${hat_dir}/tools/predict.py --config ${config} --stage int_infer

# 模型编译
# python3 -W ignore ${hat_dir}/tools/deploy/compile_perf.py --config ${config}
# 注：HAT支持两种编译模式:
# 一种是在config里写compile_cfg，通过tools/compile_perf.py启动；
# 另一种是保存一个pt，然后通过tools/compile_standalone.py启动。
# 但实际上两个都有点问题，不太好用。所以暂时通过如下方式单独运行：
python3 -W ignore ${hat_dir}/projects/planning/tools/compile.py