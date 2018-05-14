#!/bin/bash
# The startup script for competition

python main.py preproc -i $1 -o data.tfrecord -d dict.data && python main.py predict -n bilstm-cosine -d dict.data -i data.tfrecord | sort -k1n > $2
