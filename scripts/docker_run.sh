#!/usr/bin/env bash

source scripts/check_cwd.sh
source scripts/docker.env

docker run \
  --name=${REPO_NAME} \
  --rm \
  --gpus=all \
  --ipc=host \
  --env CUDA_VISIBLE_DEVICES \
  --volume $PWD:/workspace/ \
  --publish ${PORT}:8000 \
  ${DOCKER_IMAGE_GIT_HASH}
