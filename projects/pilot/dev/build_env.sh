# aidi env setting
auto_token="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjIyOTQyODY1MDEsIlRva2VuVHlwZSI6ImxkYXAiLCJVc2VyTmFtZSI6InBpbG90LnJ1bm5lciIsIk9yZ2FuaXphdGlvbiI6InJlZ3VsYXItZW5naW5lZXIiLCJPcmdhbml6YXRpb25JRCI6MX0.eT42x-fpo7SglwFtMMxs67FUYElf51ZjTeK_mqzOCAMLqxDIx8ikUMh7X07fS6NgNv6pJ5tOZZ-N9L2v0wL_h5tDIy8DsKiux_IeSgzmjXYWWzD7n6RLoLmozdpCyakBIoLE32SiVI-eV_XkEJrAtZYVGpUALmwikRcL2A-hlTzcVzE0hcOM2M3n6S7SNKAINp4EqtYVY--sxvVlPuO8m0yx3w-yQjridGenIuGFKltpLBhmZn5nj2rgDr0sg8mnFsCgbxN8BmAv5Tq1SkpF6CFHfhVCkOlGnYPMFfVWMpybCnzrjMdSu8GnQOUYAOs_fVLZxNODT3VNtxjFTa1Aqg"
export USER_TOKEN=${auto_token}

## setup gallery-cli
# curl http://gallery.hobot.cc/client/setup.sh?mode=single | bash
mkdir -p ~/.local/bin/
curl http://gallery.hobot.cc/download/devops/gallery/gallery-cli/project/release/$PLATFORM/x86_64/general/basic/0.1.9-4-g4fa9f3b/gallery-cli-0.1.9-4-g4fa9f3b -o ~/.local/bin/gallery-cli-0.1.9-4-g4fa9f3b
chmod +x ~/.local/bin/gallery-cli-0.1.9-4-g4fa9f3b
ln -s ~/.local/bin/gallery-cli-0.1.9-4-g4fa9f3b ~/.local/bin/gallery-cli

source ~/.bashrc

# install deps
pip_ext="-i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc"

pip3 install --user --use-deprecated=legacy-resolver -r requirements/develop.txt ${pip_ext}

# install hdflow deps
aidisdk config --token $auto_token --endpoint "http://aidi.hobot.cc"
python3 -m hatbc.auto_dp.cli configure --token $auto_token --cluster-app-id xxx --cluster-app-key xxx --check
evalcli configure --host http://model.aidi.hobot.cc --token $auto_token --project_id PDT20220001
reportcli configure --host http://model.aidi.hobot.cc --token $auto_token  --project_id PDT20220001

# check env
pip3 list
