

aidi_token=eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjIyNzgxNTE2NjAsIlRva2VuVHlwZSI6ImxkYXAiLCJVc2VyTmFtZSI6Inl1ZTAxLnNoaSIsIk9yZ2FuaXphdGlvbiI6InJlZ3VsYXItZW5naW5lZXIiLCJPcmdhbml6YXRpb25JRCI6MX0.unDV0Pg-bbWHTUXdx8C8mvHkTufC73A1nZAZBkYqPFizXDTHOpUjGWiurmsp3N9B1cw6igFeUXVNmf6Kfdag-Qkfwam2YB3dq5GHBIM8-b94l5ECEvZil5z6PHJ1UjaGwI919leqWDHKIdX8nlvsds_IO7_9Phx4dhlnUjBQiAFqUhKVuJasilZ7w0BxG3RD_Zm_OXdH4DEuNYHtI2Q0UZ8TERzR5CYLdXfJZj9LTVX51aBaa5Zy6ZegfJw0Vdcug_UhgRneh4BXB2NCVpxwFV_Q-R-1_lmTXqaWMHdIPjiGHfBwZwKj2BMYdHWWY-lDl0nmSju1CANi8wh28q5vng
password=175648wASD.
hitc init --token ${aidi_token}
hitc who


python3 -m hatbc.auto_dp.cli configure \
    --token ${aidi_token} \
    --cluster-app-id XqdZPBeqtL \
    --cluster-app-key rRZqiRPiLgcJIQUwtUxu \
    --check


python3 -m fordring.atlassian.cli configure \
    --jira https://jira.hobot.cc:8443 \
    --confluence http://wiki.hobot.cc \
    --username yue01.shi \
    --password ${password}


python3 -m hatbc.horizon_label_platform.cli configure \
    --host http://biaozhu.horizon.ai \
    --token ${aidi_token}


python3 -m fordring.smb.cli configure \
        --username yue01.shi \
        --password ${password}



