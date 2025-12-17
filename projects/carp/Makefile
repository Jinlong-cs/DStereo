USER_BASE=

carp-extra-env:
	cd plugins/; \
	git clone git@gitlab.hobot.cc:ptd/algorithm/asr/wenet.git wenet; \
	cd ..; pip3 install ${USER_BASE} torchaudio==0.9.1 -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc; \
	pip3 install ${USER_BASE} h5py==3.1.0 -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc

carp-bucket:
	cd ../../;  python3 tools/prepare_bucket.py --bucket "J2MM" --mount --create-link