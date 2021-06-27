#!/usr/bin/env bash
source env.sh

batch=2
d=512
interval_scale=0.5
inverse_depth=True
ckpt=./checkpoints/aa-rmvsnet_model.ckpt

CUDA_VISIBLE_DEVICES=0 python -u eval.py \
        --dataset=data_eval_transform_large \
        --batch_size=${batch} \
        --inverse_depth=${inverse_depth} \
        --numdepth=$d \
        --interval_scale=$interval_scale \
        --max_h=512 \
        --max_w=960 \
        --image_scale=1.0 \
        --testpath=$TP_TESTING \
        --testlist=lists/tp_list_int.txt \
        --loadckpt=$ckpt \
        --outdir=./outputs_tnt

