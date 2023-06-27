#!/bin/bash

if [ "$0" = "${BASH_SOURCE}" ]; then
  echo "This script must be sourced, not executed: 'source scripts/activate.sh' or '. scripts/activate.sh'"
else
  conda activate totalsegmenter
fi
