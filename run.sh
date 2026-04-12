#!/bin/bash
set -ex
rm -f graph.png
python train.py
docker build -t naive-sgd-draw .
docker run --rm -v "$PWD:/app" naive-sgd-draw
