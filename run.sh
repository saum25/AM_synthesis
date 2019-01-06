#!/bin/bash

clear all
rm -rf results
mkdir results

for lr in 5.0
do
 echo learning rate: $lr
 mkdir results/lr_$lr
 mkdir results/lr_$lr/examples
 python activation_maximisation.py --init_lr $lr --n_iters 10000 --output_dir 'results' --reg_param 1
done
