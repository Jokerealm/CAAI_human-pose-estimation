#!/bin/bash

nohup python main_hnn.py \
    --frames 27 \
    --batch_size 1024 \
    --nepoch 50 \
    --lr 0.0002 \
    --dataset h36m \
    > log/output.log 2>&1 &
    # --previous_dir checkpoint/1213_2352_01_27/model_32_2719.pth \