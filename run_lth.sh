#!/bin/bash

# BEST SGD CONFIGURATION FROM BEHNAM PAPER
best_momentum=0.9
best_wd=0.0002
# best_dropout=false

CUDA_VISIBLE_DEVICES=1 python main_lth.py --epochs 400 \
	--use-sgd --momentum ${best_momentum} --weight-decay ${best_wd} \
	--full-train --save-model \
	--prune-iterations 10 --prune-percent 20
