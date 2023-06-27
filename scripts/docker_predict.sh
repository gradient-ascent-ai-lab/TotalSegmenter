#!/usr/bin/env bash

source scripts/check_cwd.sh
source scripts/docker.env

export STATISTICS=${1:-'true'}
export RADIOMICS=${2:-'true'}
export COMMAND="totalsegmenter predict-batch"

if [[ ${STATISTICS} == 'true' ]]; then
  COMMAND="${COMMAND} --statistics"
fi
if [[ ${RADIOMICS} == 'true' ]]; then
  COMMAND="${COMMAND} --radiomics"
fi

echo
echo "Running command: ${COMMAND}"
echo

docker run \
  --name=${REPO_NAME} \
  --rm \
  --gpus=all \
  --ipc=host \
  --env CUDA_VISIBLE_DEVICES \
  --volume $PWD:/workspace/ \
  --publish ${PORT}:8000 \
  ${DOCKER_IMAGE_GIT_HASH} \
  ${COMMAND}
