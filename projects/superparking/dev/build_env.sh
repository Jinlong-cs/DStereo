# aidi env setting
auto_token="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjIzMTEyMzczMTMsIlRva2VuVHlwZSI6ImxkYXAiLCJVc2VyTmFtZSI6InBhcmtpbmcucHVibGljIiwiRW1haWwiOiIiLCJUZW5hbnQiOiJyZWd1bGFyLWVuZ2luZWVyIiwiVGVuYW50SUQiOjEsIk9yZ2FuaXphdGlvbiI6InJlZ3VsYXItZW5naW5lZXIiLCJPcmdhbml6YXRpb25JRCI6MX0.Jw6XbqtckH9d2MewXFG23N4xYckNpX5SwSqrGKih_MpvwqHtZanIYUkqodcHbabcjeHgLkhGdi0A09WTj2fofDpkgnT2AjJ3xJdOtCPvhFOevba_HocR5rtwNuktK7Q7tkfKMaXbA2bf_v9yZemYqXkTKj21YvkkLIUjxP0sb_u8CtA75bhL7sVs8Kb6nRZeTd0M_kRUTqZBO4LGzLSughanbV2efrBBg5-bZItPY1njrbxr9WrqzgKdovh59NnQ_KELK9u34BRIpd1M5UiiaqjbOrtdKz_JSHJ8HFgXIqsjABjJZDVRjRg-h44FK388RqMwZy33_6Aaga695F4zHQ"
export USER_TOKEN=${auto_token}

echo ===== prefore installation env =====
pip3 list
echo ===== prefore installation env =====

# install deps
pip_ext="-i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc"

make run-env-cu111-torch1102
pip3 install --use-deprecated=legacy-resolver -r projects/superparking/requirements.patch ${pip_ext}
pip3 install --use-deprecated=legacy-resolver https://pypi.hobot.cc/hobot-local/packages/matrix_gluon/0.5.0b202211071558+3c658c4/matrix_gluon-0.5.0b202211071558+3c658c4-py3-none-any.whl#md5=4501e21f58870b8e9e0cccfd1c4fbcce "opencv-python >= 4.6"  ${pip_ext}
pip3 install --use-deprecated=legacy-resolver mxnet-horizon-cu111==1.5.1.2.4 "opencv-python >= 4.6"  ${pip_ext}
pip3 install --use-deprecated=legacy-resolver numpy==1.23.5  ${pip_ext}
pip3 install --use-deprecated=legacy-resolver scikit-learn==0.22  ${pip_ext}
pip3 install --use-deprecated=legacy-resolver protobuf==3.20.3  ${pip_ext}
pip3 install --use-deprecated=legacy-resolver sphinx==4.0.2  ${pip_ext}

echo ===== after installation env =====
pip3 list
echo ===== after installation env =====
