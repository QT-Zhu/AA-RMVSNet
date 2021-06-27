#!/bin/bash
source env.sh

# --------test tanks and temples----------

testlist=./lists/tp_list_int.txt
outdir=./outputs_tnt/checkpoints_aa-rmvsnet_model.ckpt
test_dataset=tnt
python fusion.py --testpath=$TP_TESTING \
                     --testlist=$testlist \
                     --outdir=$outdir \
                     --test_dataset=$test_dataset

# -----------test dtu------------------

# testlist=./lists/dtu/test.txt
# outdir=./outputs_dtu/checkpoints_aa-rmvsnet_model.ckpt
# test_dataset=dtu
# python fusion.py --testpath=$DTU_TESTING \
#                     --testlist=$testlist \
#                     --outdir=$outdir \
#                     --test_dataset=$test_dataset 
              
