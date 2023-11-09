#!/bin/bash
source activate unilite
export DETETRON2_DATASETS=/15127306268
cd /13559197192/wyh/UNIReID/
pip install -U openmim
mim install mmcv-full
pip install scikit-learn
python train_net.py --config_file /13559197192/wyh/UNIReID/configs/MSVR310/vit_top_re_384_12.yml
