#!/usr/bin/env bash

set -euo pipefail

source scripts/check_cwd.sh
source scripts/docker.env

docker build --progress=plain --tag ${DOCKER_IMAGE_GIT_HASH} --tag ${DOCKER_IMAGE_GIT_LATEST} .
