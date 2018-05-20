#!/bin/bash
# The startup script for competition

python main.py preproc -i $1 -o test --no-fit && \
	python main.py predict -n bilstm -f test.tfrecord > $2
