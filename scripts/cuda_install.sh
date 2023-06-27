#!/bin/bash

sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y git unzip gcc linux-headers-$(uname -r) libsm6 libxext6
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
rm cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-drivers
sudo apt-get install -y cuda cuda-drivers
