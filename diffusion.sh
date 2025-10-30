#!/bin/bash

nohup python main_diffusion.py \
    --frames 27 \
    --batch_size 1024 \
    --nepoch 100 \
    --lr 0.0005 \
    --dataset h36m \
    > log/output.log 2>&1 &