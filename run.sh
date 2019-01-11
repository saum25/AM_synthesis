#!/bin/bash

clear all

# remove all the results
rm -rf results
rm -rf params_dump*

# create results directory
mkdir results
mkdir results/Adam
mkdir results/SGD

# string variables for naming the results directory
lr_str="lr_"
rp_str="_rp_"

# watch out there is no space between '=' sign
scale_factor=0.1

for lr in 1.0 0.1 0.01 0.001 0.0001
do
	# bash doesn't support floating point arithmetic, so use bc to process
	# watch for the use of $ sign while saving the output to another variable
	# currently not used
	adaptive_reg=$(echo $lr*$scale_factor | bc -l)

	for rp in 0.1 0.01 0.001 0.0001
	do
		echo ++++++++++++++++++++learning rate: $lr regularisation paramter: $rp++++++++++++++

		mkdir results/Adam/$lr_str$lr$rp_str$rp
		mkdir results/Adam/$lr_str$lr$rp_str$rp/examples
		python activation_maximisation.py --output_dir 'results/Adam' --init_lr $lr --reg_param $rp --n_iters 10000 --param_file 'params_dump_Adam.txt'

		mkdir results/SGD/$lr_str$lr$rp_str$rp
		mkdir results/SGD/$lr_str$lr$rp_str$rp/examples
		python activation_maximisation.py --output_dir 'results/SGD' --optimizer 'SGD' --init_lr $lr --reg_param $rp --n_iters 10000 --param_file 'params_dump_SGD.txt'
		
	done
done
