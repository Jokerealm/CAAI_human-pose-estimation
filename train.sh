#!/bin/bash

nohup python main_agformer.py \
    --frames 27 \
    --batch_size 1024 \
    --nepoch 100 \
    --lr 0.0002 \
    --dataset h36m \
    > log/output_agformer.log 2>&1 &
