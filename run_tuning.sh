#!/bin/bash

rm -rf runs/*

for reg1 in 2e-5 1e-5 5e-6 2e-6 1e-6 
do
	pids=
	for reg in 2e-5 1e-5 5e-6 2e-6 1e-6 
	do
		CUDA_VISIBLE_DEVICES=0 python main.py --epochs 400 \
			--reg1 ${reg1} --reg ${reg} --beta 50 --dropout &
		pids+=" $!"
	done
	wait $pids
done

for reg1 in 2e-5 1e-5 5e-6 2e-6 1e-6 
do
	pids=
	for reg in 2e-5 1e-5 5e-6 2e-6 1e-6 
	do
		CUDA_VISIBLE_DEVICES=0 python main.py --epochs 400 \
			--reg1 ${reg1} --reg ${reg} --beta 50 &
		pids+=" $!"
	done
	wait $pids
done

