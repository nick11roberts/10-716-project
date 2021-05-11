#!/bin/bash

rm -rf runs/*

# BEST SGD CONFIGURATION FROM BEHNAM PAPER
best_momentum=0.9
best_wd=0.0002 # Dont use weight decay
# best_dropout=false


CUDA_VISIBLE_DEVICES=0 python main_lth.py --epochs 400 \
	--use-sgd --momentum ${best_momentum} \
	--full-train --save-model \
	--prune-iterations 10 --prune-percent 20 \
	--offline-eval &



# BEST beta=50 CONFIGURATION FROM BEHNAM PAPER
best_reg1=2e-5
best_reg=5e-6
# best_dropout=false

CUDA_VISIBLE_DEVICES=1 python main_lth.py --epochs 4000 \
	--reg1 ${best_reg1} --reg ${best_reg} --beta 50 --full-train --save-model \
	--offline-eval &

