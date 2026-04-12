#!/bin/bash
set -ex
rm -f graph.png
docker build -t naive-sgd .
docker run --rm -v "$PWD:/app" naive-sgd python train2.py
docker run --rm -v "$PWD:/app" naive-sgd python draw.py
