#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/mnt/cfs/junbo/repository/IS-Fusion

TASK_DESC=$1
PORT=$((8000 + RANDOM %57535))


CONFIG=configs/isfusion/isfusion_0075voxel.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8 --master_port ${PORT} $(dirname "$0")/train.py --launcher pytorch $CONFIG \
--extra_tag $TASK_DESC




