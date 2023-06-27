#!/usr/bin/env bash

export SOURCE_PATH=${1:-${SOURCE_PATH}}
export TARGET_PATH=${2:-${TARGET_PATH}}
export BUCKET_NAME=${3:-${BUCKET_NAME}}
export DIRECTION=${4:-"push"}

echo
echo "SOURCE_PATH: ${SOURCE_PATH}"
echo "TARGET_PATH: ${TARGET_PATH}"
echo "BUCKET_NAME: ${BUCKET_NAME}"
echo "DIRECTION: ${DIRECTION}"
echo

# if source is a file
if [ -f "${SOURCE_PATH}" ]; then
  if [ "${DIRECTION}" == "pull" ]; then
    export CMD="gsutil cp gs://${BUCKET_NAME}/${SOURCE_PATH} ${TARGET_PATH}"
  else
    export CMD="gsutil cp ${SOURCE_PATH} gs://${BUCKET_NAME}/${TARGET_PATH}"
  fi
else
  if [ "${DIRECTION}" == "pull" ]; then
    export CMD="gsutil -m rsync -r gs://${BUCKET_NAME}/${SOURCE_PATH} ${TARGET_PATH}"
  else
    export CMD="gsutil -m rsync -r ${SOURCE_PATH} gs://${BUCKET_NAME}/${TARGET_PATH}"
  fi
fi

if ! command -v gsutil &> /dev/null; then
  echo "gsutil could not be found, please install it or edit this script to use another tool"
  exit
fi

if [ "${DIRECTION}" == "pull" ]; then
  echo "Pulling from gs://${BUCKET_NAME}/${SOURCE_PATH} to ${TARGET_PATH}"
  ${CMD}
  exit 0
fi

echo "Pushing from ${SOURCE_PATH} to gs://${BUCKET_NAME}/${TARGET_PATH}"
${CMD}
