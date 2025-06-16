#!/bin/bash

layers=(1 2 3)
hiddens=(64 256 1024)
dropouts=(0.0 0.2 0.4)

for layer in "${layers[@]}"
do
	for hidden in "${hiddens[@]}"
	do
		for dropout in "${dropouts[@]}"
		do
			python train.py data/shakespeare.txt --model lstm --device mps --n_epochs 100 --hidden_size $hidden --n_layers $layer --dropout $dropout
		done
	done
done
