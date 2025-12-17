set -e

export PATH=${HOME}/bin:`pwd`:/root/.local/bin:/home/cicd/.local/bin:/home/cicd/bin:$PATH
export PYTHONPATH=`pwd`:$PYTHONPATH
mkdir ${HOME}/bin

# parsing tag from branch
# E.g. "refs/tags/xxx" to "xxx"
tag=${gitlabTargetBranch}
echo ${gitlabTargetBranch}


if [ ${PILOTPATH} ]
then
    pilot_path=${PILOTPATH}
else
    pilot_path="projects"
fi

tag=(${tag//\// })
tag=${tag[2]}
echo ${tag}

# get ltc from tag
# E.g. "pilot-as33-v0.0.1" to "as33"
tag_split=(${tag//-/ })

if [ ${#tag_split[*]} -eq 3 ]
then
    ltc=${tag_split[1]}
    version=${tag_split[2]}
else
    echo "Unsupported project ltc tag format!"
    echo $tag
    exit 1
fi

# build env
## setup gallery-cli
curl http://gallery.hobot.cc/client/setup.sh?mode=single | bash

# aidi env setting
auto_token="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjIyOTQyODY1MDEsIlRva2VuVHlwZSI6ImxkYXAiLCJVc2VyTmFtZSI6InBpbG90LnJ1bm5lciIsIk9yZ2FuaXphdGlvbiI6InJlZ3VsYXItZW5naW5lZXIiLCJPcmdhbml6YXRpb25JRCI6MX0.eT42x-fpo7SglwFtMMxs67FUYElf51ZjTeK_mqzOCAMLqxDIx8ikUMh7X07fS6NgNv6pJ5tOZZ-N9L2v0wL_h5tDIy8DsKiux_IeSgzmjXYWWzD7n6RLoLmozdpCyakBIoLE32SiVI-eV_XkEJrAtZYVGpUALmwikRcL2A-hlTzcVzE0hcOM2M3n6S7SNKAINp4EqtYVY--sxvVlPuO8m0yx3w-yQjridGenIuGFKltpLBhmZn5nj2rgDr0sg8mnFsCgbxN8BmAv5Tq1SkpF6CFHfhVCkOlGnYPMFfVWMpybCnzrjMdSu8GnQOUYAOs_fVLZxNODT3VNtxjFTa1Aqg"
export USER_TOKEN=${auto_token}

## setup base env
source ${pilot_path}/pilot/dev/build_env.sh

## update env according to params
python3 ${pilot_path}/pilot/dev/publish/update_env.py --sub-project ${ltc}  --publish-version ${version}

# run pipeline
python3 ${pilot_path}/pilot/dev/publish/run_pipeline.py --sub-project ${ltc}  --publish-version ${version}
## --hbdk-verify-perf not ready
