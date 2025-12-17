#!/usr/bin/env bash
set -e

user_name=$1
user_uid=$2

useradd ${user_name} -u ${user_uid}
usermod -a -G root ${user_name}
chmod g+w -R release