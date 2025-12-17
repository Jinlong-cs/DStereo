#!/usr/bin/env bash

set -e

cp ../../dev/dockerfiles/python38/aidi_runtime_torch1102_cu111.Dockerfile $1

sed '6,8d' -i $1
sed '$a\ADD release ./release' -i $1

