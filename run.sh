#!/bin/bash
set -e

if [ "$1" = "b" ]; then
  set -x
  pushd docker
    docker build -t network .
  popd
  set +x
fi

set -x
docker run -it -v`pwd`:/w -w /w network python draw.py
