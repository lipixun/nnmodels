#!/bin/bash
# The startup script for competition

python main.py preproc -i $1 -o test -d dict.data --no-fit && \
	python main.py predict -n bilstm -d dict.data -f test.tfrecord > $2
