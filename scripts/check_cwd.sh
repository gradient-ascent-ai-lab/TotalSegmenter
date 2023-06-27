#!/usr/bin/env bash

if [ ! -d .git ]; then
  echo "This script should be run from the root of the git repository"
  exit 1
fi
