#!/bin/bash
set -ex
N=${1:-1}
docker build -t naive-sgd .
docker run -it --rm -v "$PWD:/app" naive-sgd python train${N}.py
docker run -it --rm -v "$PWD:/app" naive-sgd python draw.py
