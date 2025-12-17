# Makefile used to execute some commands locally

# set USER_BASE=--user if you meant to use 'pip3 install --user'
USER_BASE=
PIP_EXT:=-i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc

basic-data:
	export PYTHONPATH=`pwd`:${PYTHONPATH}; \
	python3 tools/prepare_bucket.py --bucket "HDLTAlgorithm" --mount --create-link


clean-basic-data:
	rm -rf tmp_data; \
	rm -rf tmp_pretrained_models \
	rm -rf tmp_orig_data


env:
	./dev/prepare_develop_env.sh ${USER_BASE}


lint:
	./dev/prepare_develop_env.sh ${USER_BASE}; \
	pre-commit run --all-files


isort:
	isort .


black:
	black . -l 79


flake8:
	flake8 .


pydocstyle:
	pydocstyle --match-dir='(?!test|project).*'


doc:
	export PYTHONPATH=`pwd`:${PYTHONPATH}; \
	cd docs && make html; \
	cd ..


clean-doc:
	cd docs && make clean; \
	cd ..


run-basic-env-cpu-torch201:
	pip3 install ${USER_BASE} torch==2.0.1+cpu -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install ${USER_BASE} torchvision==0.15.2+cpu -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install --use-deprecated=legacy-resolver ${USER_BASE} -r requirements.txt ${PIP_EXT}; \
        pip3 install ${USER_BASE} -U hbdk-internal ${PIP_EXT}; \
	pip3 install ${USER_BASE} -U horizon-plugin-pytorch -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/cpu/torch201 --trusted-host art-internal.hobot.cc ${PIP_EXT}; \
        pip3 install ${USER_BASE} -U horizon-plugin-profiler ${PIP_EXT}

run-basic-env-cpu:
	pip3 install ${USER_BASE} torch==1.13.0+cpu -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install ${USER_BASE} torchvision==0.14.0+cpu -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install --use-deprecated=legacy-resolver ${USER_BASE} -r requirements.txt ${PIP_EXT}; \
        pip3 install ${USER_BASE} -U hbdk-internal ${PIP_EXT}; \
	pip3 install ${USER_BASE} -U horizon-plugin-pytorch -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/cpu/torch1130 --trusted-host art-internal.hobot.cc ${PIP_EXT}; \
        pip3 install ${USER_BASE} -U horizon-plugin-profiler ${PIP_EXT}


run-basic-env-cu116:
	pip3 install ${USER_BASE} torch==1.13.0+cu116 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install ${USER_BASE} torchvision==0.14.0+cu116 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install --use-deprecated=legacy-resolver ${USER_BASE} -r requirements.txt ${PIP_EXT}; \
        pip3 install ${USER_BASE} -U hbdk-internal ${PIP_EXT}; \
	pip3 install ${USER_BASE} -U horizon-plugin-pytorch -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/cu116/torch1130 --trusted-host art-internal.hobot.cc ${PIP_EXT}; \
        pip3 install ${USER_BASE} -U horizon-plugin-profiler ${PIP_EXT}

run-basic-env-cu118:
	pip3 install ${USER_BASE} torch==2.0.1+cu118 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install ${USER_BASE} torchvision==0.15.2+cu118 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install --use-deprecated=legacy-resolver ${USER_BASE} -r requirements.txt ${PIP_EXT}; \
        pip3 install ${USER_BASE} -U hbdk-internal ${PIP_EXT}; \
	pip3 install ${USER_BASE} -U horizon-plugin-pytorch -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/cu118/torch201 --trusted-host art-internal.hobot.cc ${PIP_EXT}; \
        pip3 install ${USER_BASE} -U horizon-plugin-profiler ${PIP_EXT}

run-env-cpu-torch201:
	pip3 install ${USER_BASE} torch==2.0.1+cpu -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install ${USER_BASE} torchvision==0.15.2+cpu -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install ${USER_BASE} torchaudio==2.0.2+cpu -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install --use-deprecated=legacy-resolver ${USER_BASE} -r requirements.txt ${PIP_EXT}; \
        pip3 install ${USER_BASE} -U hbdk-internal ${PIP_EXT}; \
	pip3 install ${USER_BASE} -U horizon-plugin-pytorch -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/cpu/torch201 --trusted-host art-internal.hobot.cc ${PIP_EXT}; \
        pip3 install ${USER_BASE} -U horizon-plugin-profiler ${PIP_EXT}

run-env-cpu:
	pip3 install ${USER_BASE} torch==1.13.0+cpu -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install ${USER_BASE} torchvision==0.14.0+cpu -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install ${USER_BASE} torchaudio==0.13.0+cpu -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install --use-deprecated=legacy-resolver ${USER_BASE} -r requirements.txt ${PIP_EXT}; \
        pip3 install ${USER_BASE} -U hbdk-internal ${PIP_EXT}; \
	pip3 install ${USER_BASE} -U horizon-plugin-pytorch -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/cpu/torch1130 --trusted-host art-internal.hobot.cc ${PIP_EXT}; \
        pip3 install ${USER_BASE} -U horizon-plugin-profiler ${PIP_EXT}

run-env-cu118:
	pip3 install ${USER_BASE} torch==2.0.1+cu118 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install ${USER_BASE} torchvision==0.15.2+cu118 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install ${USER_BASE} torchaudio==2.0.2+cu118 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install ${USER_BASE} pytorch3d -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/pytorch3d/cu118/torch201/ --trusted-host art-internal.hobot.cc ${PIP_EXT}; \
	pip3 install --use-deprecated=legacy-resolver ${USER_BASE} -r requirements.txt ${PIP_EXT}; \
        pip3 install ${USER_BASE} -U hbdk-internal ${PIP_EXT}; \
	pip3 install ${USER_BASE} -U horizon-plugin-pytorch -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/cu118/torch201 --trusted-host art-internal.hobot.cc ${PIP_EXT}; \
        pip3 install ${USER_BASE} -U horizon-plugin-profiler ${PIP_EXT}

run-env-cu116:
	pip3 install ${USER_BASE} torch==1.13.0+cu116 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install ${USER_BASE} torchvision==0.14.0+cu116 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install ${USER_BASE} torchaudio==0.13.0+cu116 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install ${USER_BASE} torchdynamo==1.13.0 ${PIP_EXT}; \
	pip3 install ${USER_BASE} pytorch3d -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/pytorch3d/cu116/torch1130/ --trusted-host art-internal.hobot.cc ${PIP_EXT}; \
	pip3 install --use-deprecated=legacy-resolver ${USER_BASE} -r requirements.txt ${PIP_EXT}; \
        pip3 install ${USER_BASE} -U hbdk-internal ${PIP_EXT}; \
	pip3 install ${USER_BASE} -U horizon-plugin-pytorch -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/cu116/torch1130 --trusted-host art-internal.hobot.cc ${PIP_EXT}; \
        pip3 install ${USER_BASE} -U horizon-plugin-profiler ${PIP_EXT}


run-env-cpu-torch1102:
	pip3 install ${USER_BASE} torch==1.10.2+cpu -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install ${USER_BASE} torchvision==0.11.3+cpu -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install ${USER_BASE} torchaudio==0.10.2+cpu -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install --use-deprecated=legacy-resolver ${USER_BASE} -r requirements.txt ${PIP_EXT}; \
        pip3 install ${USER_BASE} -U hbdk-internal ${PIP_EXT}; \
	pip3 install ${USER_BASE} -U horizon-plugin-pytorch -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/cpu/torch1102 --trusted-host art-internal.hobot.cc ${PIP_EXT}; \
        pip3 install ${USER_BASE} -U horizon-plugin-profiler ${PIP_EXT}

run-env-cu102-torch1102:
	pip3 install ${USER_BASE} torch==1.10.2+cu102 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install ${USER_BASE} torchvision==0.11.3+cu102 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install ${USER_BASE} torchaudio==0.10.2+cu102 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install ${USER_BASE} pytorch3d -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/pytorch3d/cu102/torch1102/ --trusted-host art-internal.hobot.cc ${PIP_EXT}; \
	pip3 install --use-deprecated=legacy-resolver ${USER_BASE} -r requirements.txt ${PIP_EXT}; \
        pip3 install ${USER_BASE} -U hbdk-internal ${PIP_EXT}; \
	pip3 install ${USER_BASE} -U horizon-plugin-pytorch -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/cu102/torch1102 --trusted-host art-internal.hobot.cc ${PIP_EXT}; \
        pip3 install ${USER_BASE} -U horizon-plugin-profiler ${PIP_EXT}


run-env-cu111-torch1102:
	pip3 install ${USER_BASE} torch==1.10.2+cu111 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install ${USER_BASE} torchvision==0.11.3+cu111 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install ${USER_BASE} torchaudio==0.10.2+cu111 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${PIP_EXT}; \
	pip3 install ${USER_BASE} pytorch3d -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/pytorch3d/cu111/torch1102/ --trusted-host art-internal.hobot.cc ${PIP_EXT}; \
	pip3 install --use-deprecated=legacy-resolver ${USER_BASE} -r requirements.txt ${PIP_EXT}; \
        pip3 install ${USER_BASE} -U hbdk-internal ${PIP_EXT}; \
	pip3 install ${USER_BASE} -U horizon-plugin-pytorch -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/cu111/torch1102 --trusted-host art-internal.hobot.cc ${PIP_EXT}; \
        pip3 install ${USER_BASE} -U horizon-plugin-profiler ${PIP_EXT}


collect-env:
	python3 -m hat.utils.collect_env


serial-unit-test:
	export PYTHONPATH=`pwd`:${PYTHONPATH}; \
	pytest -m "serial_task" --reruns 3 --reruns-delay 5 -s -x tests/unit_tests --cov hat --cov-report term-missing --cov-report xml:./cov.xml; \


unit-test:
	export PYTHONPATH=`pwd`:${PYTHONPATH}; \
	pytest -m "not serial_task" --reruns 3 --reruns-delay 5 --rerun-except Timeout -s -x tests/unit_tests -n 3 --cov hat --cov-report term-missing --cov-report xml:./cov.xml

intergration-tests:
	export PYTHONPATH=`pwd`:${PYTHONPATH}; \
	pytest -k "not test_aidiexp_run" tests/intergration_tests --reruns 3 --reruns-delay 5 -s -x -n 3 --cov hat --cov-report term-missing --cov-report xml:./cov.xml
	pytest tests/intergration_tests/test_aidiexp_run.py --reruns 3 --reruns-delay 5 -s -x -n 1 --cov hat --cov-report term-missing --cov-report xml:./cov.xml

wheel:
	python3 setup.py sdist bdist_wheel; \
	ls dist


clean-wheel:
	rm -rf dist

set_hobot_source:
	pip3 config ${USER_BASE} set global.index-url https://pypi.hobot.cc/simple; \
	pip3 config ${USER_BASE} set global.trusted-host pypi.hobot.cc; \
	pip3 config ${USER_BASE} set extra.index-url https://pypi.hobot.cc/hobot-local/simple; \
