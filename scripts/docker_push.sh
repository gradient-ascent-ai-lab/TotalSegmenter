#!/usr/bin/env bash

source scripts/check_cwd.sh
source scripts/docker.env

docker push ${DOCKER_IMAGE_GIT_HASH}
docker push ${DOCKER_IMAGE_GIT_LATEST}
