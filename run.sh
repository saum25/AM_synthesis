#!/bin/bash

clear all

# remove all the results
rm -rf results
rm -rf param_dump.txt

mkdir results

# string variables for naming the results directory
lr_str="lr_"
rp_str="_rp_"

# watch out there is no space between '=' sign
scale_factor=0.1

for lr in 1.0 0.1 0.01 0.001 0.0001
do
	# bash doesn't support floating point arithmetic, so use bc to process
	# watch for the use of $ sign while saving the output to another variable
	adaptive_reg=$(echo $lr*$scale_factor | bc -l)

	for rp in 0.1 0.01 0.001 0.0001 $adaptive_reg
	do
		echo ++++++++++++++++++++learning rate: $lr regularisation paramter: $rp++++++++++++++
		mkdir results/$lr_str$lr$rp_str$rp
		mkdir results/$lr_str$lr$rp_str$rp/examples
		python activation_maximisation.py --output_dir 'results' --init_lr $lr --reg_param $rp --n_iters 5
	done
done
