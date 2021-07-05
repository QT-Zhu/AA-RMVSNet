#!/bin/bash
source env.sh

testlist=./lists/dtu/test.txt
outdir=./outputs_dtu/checkpoints_model_release.ckpt
test_dataset=dtu
python fusion.py --testpath=$DTU_TESTING \
                    --testlist=$testlist \
                    --outdir=$outdir \
                    --test_dataset=$test_dataset 
              
