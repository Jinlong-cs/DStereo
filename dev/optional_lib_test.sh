set -e

export PATH=/root/.local/bin:/home/cicd/.local/bin/:$PATH

# prepare hat
export PYTHONPATH=$(pwd):${PYTHONPATH}

# uninstall 

pip3 uninstall -y aidisdk hatbc

# run tests
echo "----------------run stable test with pytest----------------"

pytest -s -x tests/intergration_tests/test_aidiexp_run.py

echo "----------------test success!-----------------------"

# reinstall
pip_ext="-i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc"
pip3 install --use-deprecated=legacy-resolver --user -r requirements/internal.txt ${pip_ext}

