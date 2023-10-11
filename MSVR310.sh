#!/bin/bash
source activate unilite
export DETETRON2_DATASETS=/15127306268
cd /15127306268/wyh/MM/
pip install -U openmim
mim install mmcv-full
pip install scikit-learn
python train_net.py --config_file /15127306268/wyh/MM/configs/MSVR310/vit_top_re_384_12.yml
