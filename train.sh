#!/bin/bash

nohup python main.py \
    --frames 27 \
    --batch_size 1024 \
    --nepoch 50 \
    --lr 0.0002 \
    --dataset h36m \
    --train_refinement_only --refinement_lr_ratio 0.1 \
    --previous_dir checkpoint/4*a6000_baseline/model_25_2769.pth \
    > log/output.log 2>&1 &
    # --previous_dir checkpoint/4*a6000_baseline/model_25_2769.pth \