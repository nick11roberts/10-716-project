#!/bin/bash

# BEST beta=50 CONFIGURATION FROM BEHNAM PAPER
best_reg1=2e-5
best_reg=5e-6
# best_dropout=false

pids=
for iters in 1 2 4 8 16 32
do
	CUDA_VISIBLE_DEVICES=0 python main.py --epochs 400 \
		--reg1 ${best_reg1} --reg ${best_reg} --beta 50 --soft-threshold-iters ${iters} --full-train --save-model &
	pids+=" $!"
done
wait $pids

pids=
for iters in 64 128 256 512 1024 2048
do
	CUDA_VISIBLE_DEVICES=0 python main.py --epochs 400 \
		--reg1 ${best_reg1} --reg ${best_reg} --beta 50 --soft-threshold-iters ${iters} --full-train --save-model &
	pids+=" $!"
done
wait $pids