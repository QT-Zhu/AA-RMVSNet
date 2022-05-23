#!/usr/bin/env bash
source env.sh
set -e

batch=2
d=512
inverse_depth=True
ckpt=./checkpoints/model_blended_v2.ckpt

CUDA_VISIBLE_DEVICES=0 python eval.py \
        --dataset=data_eval_transform_padding \
        --batch_size=${batch} \
        --inverse_depth=${inverse_depth} \
        --numdepth=$d \
        --max_h=544 \
        --max_w=1024 \
        --image_scale=1.0 \
        --testpath=$TP_TESTING \
        --testlist=lists/tp_list_int_1024.txt \
        --loadckpt=$ckpt \
        --outdir=./outputs_tnt

CUDA_VISIBLE_DEVICES=0 python eval.py \
        --dataset=data_eval_transform_padding \
        --batch_size=${batch} \
        --inverse_depth=${inverse_depth} \
        --numdepth=$d \
        --max_h=544 \
        --max_w=960 \
        --image_scale=1.0 \
        --testpath=$TP_TESTING \
        --testlist=lists/tp_list_int_960.txt \
        --loadckpt=$ckpt \
        --outdir=./outputs_tnt


