#!/bin/bash

rm -rf runs/*

for momentum in 0 0.9 
do
	pids=
	for wd in 0 1e-4 2e-4 5e-4 1e-3
	do
		CUDA_VISIBLE_DEVICES=0 python main.py --epochs 400 \
			--use-sgd --momentum ${momentum} --weight-decay ${wd} &

		CUDA_VISIBLE_DEVICES=1 python main.py --epochs 400 \
			--use-sgd --momentum ${momentum} --weight-decay ${wd} --dropout &

		pids+=" $!"
	done
	wait $pids
done
