#!/bin/bash

srun \
--nodes=1 \
--gpus=1 \
--mem=42G \
--time=0-01:00 \
--container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.10-py3.sqsh \
--container-mounts=/ds:/ds,/netscratch:/netscratch,"`pwd`":"`pwd`" \
--container-workdir="`pwd`" \
install.sh python3 run.py -config 'config_train/test_emformer_cluster.yaml' -debug True
echo 'running'
