#!/bin/bash
# Pack the data and scripts

ModelName=$1
TFTrainerPath=$2
DictName=dict.data
OutputName=launch.tar.gz

if [[ $ModelName == "" ]]; then
	echo Require model name >&2
	exit 1
fi

if [[ $TFTrainerPath == "" ]]; then
	echo Require tftrainer path >&2
	exit 1
fi

tar -zcf $OutputName run.sh *.py $DictName outputs/$ModelName -C $TFTrainerPath tftrainer
