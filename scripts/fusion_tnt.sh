#!/bin/bash
source env.sh

testlist=./lists/tp_list_int.txt
outdir=./outputs_tnt/checkpoints_model_release.ckpt
test_dataset=tnt
python fusion_padding.py --testpath=$TP_TESTING \
                     --testlist=$testlist \
                     --outdir=$outdir \
                     --test_dataset=$test_dataset
