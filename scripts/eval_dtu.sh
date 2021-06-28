#!/usr/bin/env bash
source env.sh

batch=2
d=512
interval_scale=0.4
inverse_depth=False
ckpt=./checkpoints/model_release.ckpt

CUDA_VISIBLE_DEVICES=0 python eval.py \
        --dataset=data_eval_transform \
        --batch_size=${batch} \
        --inverse_depth=${inverse_depth} \
        --numdepth=$d \
        --interval_scale=$interval_scale \
        --max_h=600 \
        --max_w=800 \
        --image_scale=1.0 \
        --testpath=$DTU_TESTING \
        --testlist=lists/dtu/test.txt \
        --loadckpt=$ckpt \
        --outdir=./outputs_dtu

