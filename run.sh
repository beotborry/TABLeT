#!/bin/bash
# Example: sweep over test set IDs and learning rates on 4 GPUs.
# Adjust --config, --num-axis, --epochs, etc. to match your experiment.

CONFIG="configs/hcp_configs/tablet_hcp_intelligence_T256.yaml"
WANDB_NAME="tablet_hcp_intelligence_T256"

for tid in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 train.py \
        --config $CONFIG --num-axis 3 --test-set-id $tid \
        --lr 1e-06 --batch-size 4 --epochs 20 \
        --wandb-name $WANDB_NAME &

    CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 train.py \
        --config $CONFIG --num-axis 3 --test-set-id $tid \
        --lr 1e-05 --batch-size 4 --epochs 20 \
        --wandb-name $WANDB_NAME &

    CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes 1 train.py \
        --config $CONFIG --num-axis 3 --test-set-id $tid \
        --lr 1e-04 --batch-size 4 --epochs 20 \
        --wandb-name $WANDB_NAME &

    CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes 1 train.py \
        --config $CONFIG --num-axis 3 --test-set-id $tid \
        --lr 1e-03 --batch-size 4 --epochs 20 \
        --wandb-name $WANDB_NAME &

    wait
done
