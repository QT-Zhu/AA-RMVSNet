#!/usr/bin/env bash
source env.sh

batch=2
d=512
interval_scale=0.5
inverse_depth=True
ckpt=./checkpoints/model_release.ckpt

CUDA_VISIBLE_DEVICES=0 python eval.py \
        --dataset=data_eval_transform_padding \
        --batch_size=${batch} \
        --inverse_depth=${inverse_depth} \
        --numdepth=$d \
        --interval_scale=$interval_scale \
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
        --interval_scale=$interval_scale \
        --max_h=544 \
        --max_w=960 \
        --image_scale=1.0 \
        --testpath=$TP_TESTING \
        --testlist=lists/tp_list_int_960.txt \
        --loadckpt=$ckpt \
        --outdir=./outputs_tnt


