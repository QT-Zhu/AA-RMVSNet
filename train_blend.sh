#!/bin/bash
source env.sh
data=$(date +"%m%d")
batch=1
epochs=10
d=128
interval_scale=1
lr=0.001
inverse_depth=False
image_scale=0.25
view_num=7

CUDA_VISIBLE_DEVICES=0,1 python train.py  \
        --dataset=dtu_yao_blend \
        --batch_size=${batch} \
        --trainpath=$BLEND_TRAINING \
        --lr=${lr} \
        --epochs=${epochs} \
        --view_num=$view_num \
        --inverse_depth=${inverse_depth} \
        --image_scale=$image_scale \
        --max_h=576 \
        --max_w=768 \
        --trainlist=/xdata/zqt/MVS_data/dataset_low_res/training_list.txt \
        --vallist=/xdata/zqt/MVS_data/dataset_low_res/validation_list.txt \
        --testlist=/xdata/zqt/MVS_data/dataset_low_res/validation_list.txt \
        --numdepth=$d \
        --interval_scale=$interval_scale \
        --logdir=./checkpoints/${name}
