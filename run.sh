#!/bin/bash

# BEST beta=50 CONFIGURATION FROM BEHNAM PAPER
best_reg1=2e-5
best_reg=5e-6
# best_dropout=false

CUDA_VISIBLE_DEVICES=0 python main.py --epochs 4000 \
	--reg1 ${best_reg1} --reg ${best_reg} --beta 50 --full-train --save-model 

