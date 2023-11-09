#!/bin/bash
cd /13559197192/wyh/UNIReID/
pip install scikit-learn
pip install pytorch_wavelets
pip install PyWavelets
python train_net.py --config_file /13559197192/wyh/UNIReID/configs/MM/RGBNT100/vit.yml
