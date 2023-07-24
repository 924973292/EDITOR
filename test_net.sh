#!/bin/bash
source activate unilite
export DETETRON2_DATASETS=/15127306268
cd /15127306268/wyh/MM/
pip install -U openmim
mim install mmcv-full
python net_test.py --config_file /15127306268/wyh/UIS/configs/MSMT17/msmt_t2t14_res152_layer2.yml --fea_cft 6
