#!/usr/bin/env bash

export SOURCE_PATH=${1:-${SOURCE_PATH}}
export TARGET_PATH=${2:-${TARGET_PATH}}
export SSH_KEY_PATH=${3:-${SSH_KEY_PATH}}
export HOST=${4:-${HOST}}
export USER=${5:-${USER}}
export DIRECTION=${6:-"push"}

echo
echo "SOURCE_PATH: ${SOURCE_PATH}"
echo "TARGET_PATH: ${TARGET_PATH}"
echo "SSH_KEY_PATH: ${SSH_KEY_PATH}"
echo "HOST: ${HOST}"
echo "USER: ${USER}"
echo

if [ "${DIRECTION}" == "pull" ]; then
  echo "Pulling from ${HOST}:${SOURCE_PATH} to ${TARGET_PATH}"
  rsync -avz --partial --progress -e "ssh -i ${SSH_KEY_PATH}" "${USER}@${HOST}:${SOURCE_PATH}" "${TARGET_PATH}"
  exit 0
fi

echo "Pushing from ${SOURCE_PATH} to ${HOST}:${TARGET_PATH}"
rsync -avz --partial --progress -e "ssh -i ${SSH_KEY_PATH}" "${SOURCE_PATH}" "${USER}@${HOST}:${TARGET_PATH}"
