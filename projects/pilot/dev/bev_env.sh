set -e

# cu118 torch2.0.1
make run-env-cu118
make env
pip3 install -r requirements/develop.txt
pip3 install -r requirements/build.txt
pip3 install -r requirements/optional.txt

# pip3 install horizon-hdflow -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
pip3 install imutils -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
pip3 install moviepy -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
pip3 install "horizon_driving_dataset>=0.0.30" -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
pip3 install visperson -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
pip3 install "pytorch-crf>=0.7.2" -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
pip3 install fvcore -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc --use-deprecated=legacy-resolver
pip3 install pytorch3d -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/pytorch3d/cu118/torch201/ --trusted-host art-internal.hobot.cc
pip3 install "hat-sim>=1.0.1" -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
pip3 install "pathos>=0.3.0" -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
pip3 install "horizon-tat>=0.3.2" -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
pip3 install mpi4py -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
pip3 install protobuf==3.20.3 -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc --use-deprecated=legacy-resolver

# plugin2.0.0 hbdk3.43.3
pip3 install horizon-plugin-pytorch==2.0.0 -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/cu118/torch201 --trusted-host art-internal.hobot.cc
pip3 install hbdk==3.43.3 -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
pip3 install hbdk-model-verifier==3.43.3 -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
pip3 install hbdk-internal==3.43.3 -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
