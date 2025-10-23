#!/bin/bash

nohup python main_SGraAGFormer.py \
    --frames 27 \
    --batch_size 1024 \
    --nepoch 100 \
    --lr 0.0005 \
    --dataset h36m \
    --lr_decay 0.99 \
    > log/output_SGraAGFormer.log 2>&1 &
