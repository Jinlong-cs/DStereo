set -e -v
cd $(dirname $0)
config_file="./D-StereoPlus.yaml"
model_type="onnx"
hb_mapper makertbin --config ${config_file} \
                    --model-type ${model_type}
