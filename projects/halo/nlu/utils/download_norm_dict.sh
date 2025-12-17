#!/bin/bash
#Copyright: Horizon Robotic
#Function: Download nlu project to get norm dict file for protocal evaluating.

if [[ ! "$(pwd)" == *halo/nlu/utils ]]; then
    echo "Running dir for this script is wrong!Please check it!"
    exit 1
fi

if [[ -d "./nlu" ]]; then
    echo "Nlu dir already exist,removing it!"
    rm -rf ./nlu
fi

git clone git@gitlab.hobot.cc:ptd/algorithm/nlp/nlu.git

rm -f ./nlu/nlu/data/standard/dict/*.bin
