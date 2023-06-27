#!/usr/bin/env bash

set -eo pipefail

source scripts/check_cwd.sh

if ! command -v conda &> /dev/null; then
  read -p "Conda could not be found. Do you want to install it? [y/N] " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing conda..."
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    ~/miniconda3/bin/conda init bash
    ~/miniconda3/bin/conda config --set auto_activate_base false
    ~/miniconda3/bin/conda install mamba -n base -c conda-forge
    echo "Conda installed successfully."
    echo "Please restart your terminal and run this script again."
    exit
  else
    echo "Please install conda and run this script again."
    exit
  fi
fi

if command -v mamba &> /dev/null; then
  CONDA_CMD="mamba"
else
  CONDA_CMD="conda"
fi

CONDA_ENV_NAME=$(head -n 1 env.yaml | cut -d' ' -f2)
if conda env list | grep -q ${CONDA_ENV_NAME}; then
  read -p "Conda environment ${CONDA_ENV_NAME} already exists. Do you want to delete it? [y/N] " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Deleting conda environment..."
    conda env remove -n ${CONDA_ENV_NAME}
  else
    echo "Aborting setup."
    exit
  fi
fi

echo "Creating conda environment..."
${CONDA_CMD} env create -f env.yaml

echo "Activating conda environment..."
source $(conda info -q --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}

echo "Installing Python packages..."
pip install -e .[all,dev,test]

echo "Installing pre-commit hooks..."
pre-commit install
