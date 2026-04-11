#!/bin/bash
set -ex
rm -f graph.png
# python train.py
go run main.go
docker build -t naive-sgd-draw .
docker run --rm -v "$PWD:/app" naive-sgd-draw
