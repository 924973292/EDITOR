#!/bin/bash
source activate unilite
export DETETRON2_DATASETS=/15127306268
cd /15127306268/wyh/MM/
pip install -U openmim
mim install mmcv-full
python train_net.py --config_file /15127306268/wyh/MM/configs/Rotation.yml
