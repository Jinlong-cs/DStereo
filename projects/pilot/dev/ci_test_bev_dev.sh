set -e

export PATH=/root/.local/bin:/home/cicd/.local/bin/:$PATH

# prepare hat
export PYTHONPATH=$(pwd):${PYTHONPATH}

# build env
# source projects/pilot/dev/build_env.sh

aidi_host="http://aidi.hobot.cc"
ci_token="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjIyNTgwMDE5NDMsIlRva2VuVHlwZSI6ImxkYXAiLCJVc2VyTmFtZSI6InpoaWdhbmcueWFuZyIsIk9yZ2FuaXphdGlvbiI6InJlZ3VsYXItZW5naW5lZXIiLCJPcmdhbml6YXRpb25JRCI6MX0.LIISKqNmERemrWg-Q4i6Amwy7rU8zyThvrfD3EhE6MQhj772kp2R_JGHnnt6FVavrLzfiReYI9UiLlUOuN08abFfI_lqyNP1lB6-9NhnzHf-2gTmV2VVUwiSP6SPIyR1R8Wbwb_mMI8Yd1_zyoAE5r597fX8g06MchvLpRlUJheZUsgzKyMJyOTxnPtUVOHpkpel1i6gEx7S4DFBtDqNLw3oJMAdJ75DuIkTusbaW9Ko6NEWm1Jyrjz79zPRCY5w14JwiL13iEX6kpJcM7Q9b_Rumm7-QWh5hOKaluntgRlKLdlgI2d27dEE3QYQxyWvGwKnQ515LIBXyKtrP1xB1A"
aidisdk config --token ${ci_token} --endpoint ${aidi_host}

# run integration test
echo "---------------------run build integration test --------------------------"
HAT_PILOT_TEST_LEVEL=bev_dev pytest -s -x projects/pilot/tests/test_pipeline.py
echo "-------------------build integration test success!------------------------"
