config_list="
examples/hbdk4/efficientnet.py
"

echo "---------------------install hbdk4------------------------"
pip_ext="-i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc"
pip3 install hbdk4_compiler --user ${pip_ext}
echo "------------------install success!------------------"

mkdir -p tmp_models
mkdir -p tmp_models/efficientnet_cls
cp -dr /horizon-bucket/HDLTAlgorithm/models/bayes_release_models/efficientnet_cls/*.pth.tar tmp_models/efficientnet_cls/

for cfg in ${config_list};
do
  python3 tools/deploy/export_hbir.py -c ${cfg}
  python3 tools/deploy/infer_hbir.py -c ${cfg} --step-num 3
  python3 tools/deploy/compile_perf_hbir.py -c ${cfg}
done
