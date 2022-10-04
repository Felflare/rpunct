#!/bin/bash
# initial system setup
set -e

export OS=ubuntu2004

wget https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin

sudo mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/ /"
sudo apt-get update

sudo apt-get install libnvinfer8=8.4.3-1+cuda11.6
sudo apt-get install libnvinfer8-dev=8.4.3-1+cuda11.6

sudo apt-get install -y cuda-11-6


# retrieving rpunct code
git clone git@github.com:bbc/rpunct.git
cd rpunct
git checkout tp-fixing-training
mkdir training/datasets


# python setup
sudo apt-get install python3 python3-venv python3-pip

python3 -m venv .env
source .env/bin/activate

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
pip install transformers -U


# running training code
python training/prep_data.py
python training/train.py
