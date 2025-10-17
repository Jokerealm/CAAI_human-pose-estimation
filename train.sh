#!/bin/bash

nohup python main_1.py \
    --frames 27 \
    --batch_size 1024 \
    --nepoch 200 \
    --lr 0.0002 \
    --dataset h36m \
    > output_1.log 2>&1 &