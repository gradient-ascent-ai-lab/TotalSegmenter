#!/bin/bash

# if shutdown requires sudo, run this as follows:
# $ sudo su
# $ nohup ./scripts/shutdown_after.sh > /dev/null 2>&1 &

PROCESS_NAME=${1:-totalsegmenter}
PIDS="$(pgrep -f ${PROCESS_NAME})"
SHUTDOWN_CMD=${2-"sudo shutdown"}

if [ -z "${PIDS}" ]; then
  echo "No process found for ${PROCESS_NAME}"
  exit 1
fi

echo "Found processes for ${PROCESS_NAME}:"
echo "${PIDS}"
echo "Shutting down after these processes exit"

for PID in $PIDS; do
  while ps -p ${PID} > /dev/null; do sleep 10; done &
done

wait
echo "All processes have exited, initiating shutdown."
${SHUTDOWN_CMD}
