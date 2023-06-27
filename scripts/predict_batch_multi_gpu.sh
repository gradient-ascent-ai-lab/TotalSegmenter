#!/bin/bash

source ~/.bashrc
source scripts/activate.sh

export NUM_GPUS=${1:-'auto'}
export TASK_NAMES=${2:-'total'}
export STATISTICS=${3:-'true'}
export RADIOMICS=${4:-'true'}
export PREVIEW=${5:-'true'}
export GPU_IDS=${6:-'0'}

if [ "${NUM_GPUS}" = "auto" ]; then
  NUM_GPUS=$(nvidia-smi -L | wc -l)
  GPU_IDS=$(seq 0 $((${NUM_GPUS} - 1)))
fi

echo
echo "Using ${NUM_GPUS} GPUs with IDs: ${GPU_IDS}"
echo

IFS=',' read -r -a TASK_NAMES <<< "$TASK_NAMES"
echo "Running tasks:"
for TASK_NAME in "${TASK_NAMES[@]}"; do
  echo " - ${TASK_NAME}"
done

CMD="python totalsegmenter/cli.py predict-batch"
for TASK_NAME in "${TASK_NAMES[@]}"; do
  CMD="$CMD --task-name=${TASK_NAME}"
done

if [ "$STATISTICS" = "true" ]; then
  CMD="$CMD --statistics"
fi
if [ "$RADIOMICS" = "true" ]; then
  CMD="$CMD --radiomics"
fi
if [ "$PREVIEW" = "true" ]; then
  CMD="$CMD --preview"
fi

echo "Running command: '$CMD'"
echo

IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS"
for split_index in $(seq 0 $((${NUM_GPUS} - 1))); do
  export GPU_ID=${GPU_IDS[$split_index]}
  export CUDA_VISIBLE_DEVICES=${GPU_ID}
  LOG_PATH="logs/predict_batch_multi_gpu_${GPU_ID}.log"
  echo "Starting batch predict on GPU ${GPU_ID}"
  echo "Logging output to ${LOG_PATH}"
  nohup $CMD --num-splits=${NUM_GPUS} --split-index=${split_index} > ${LOG_PATH} 2>&1 &
  echo "Task PID for GPU ${GPU_ID}: $!"
  echo
done
