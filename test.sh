#!/bin/bash

python main.py \
    --test \
    --frames 27 \
    --batch_size 1024 \
    --nepoch 100 \
    --lr 0.0002 \
    --dataset h36m \
    --previous_dir checkpoint/1112_2216_27_27/model_6_2731.pth