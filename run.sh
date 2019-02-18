#!/bin/bash

# best and simple way for multiline comment
# : ' in the start and end. Watch out for any ' in between.'

clear all

# remove all the results
rm -rf results
rm -rf params_dump*

# if needed run the code for SGD optimisation
# 1 -> run for SGD as well
# 0 -> only Adam

sgd_optm=1

# create results directory
mkdir results
mkdir results/Adam

if [ $sgd_optm -eq 1 ]
then
	mkdir results/SGD
fi

# string variables for naming the results directory
lr_str="lr_"
rp_str="_rp_"

# watch out there is no space between '=' sign
#scale_factor=0.1

for neuron_idx in 0 1
do
	Adam_path=results/Adam/neuron$neuron_idx
	mkdir $Adam_path
	if [ $sgd_optm -eq 1 ]
	then
		SGD_path=results/SGD/neuron$neuron_idx
		mkdir $SGD_path
	fi

	for lr in 1.0 0.1 #0.01 0.001 0.0001
	do
		# bash does''t support floating point arithmetic, so use bc to process
		# watch for the use of $ sign while saving the output to another variable
		# currently not used
		#adaptive_reg=$(echo $lr*$scale_factor | bc -l)

		for rp in 1.0 0.1 #0.01 0.001 0.0001
		do
			echo ++++++++++++++++++++learning rate: $lr regularisation paramter: $rp++++++++++++++

			mkdir results/Adam/neuron$neuron_idx/$lr_str$lr$rp_str$rp
			mkdir results/Adam/neuron$neuron_idx/$lr_str$lr$rp_str$rp/examples
			python activation_maximisation.py --output_dir $Adam_path --init_lr $lr --reg_param $rp --n_iters 5000 --param_file 'params_dump_Adam.txt' --neuron $neuron_idx --n_out_neuron 2 --layer 'fc9'

			if [ $sgd_optm -eq 1 ]
			then
				mkdir results/SGD/neuron$neuron_idx/$lr_str$lr$rp_str$rp
				mkdir results/SGD/neuron$neuron_idx/$lr_str$lr$rp_str$rp/examples
				python activation_maximisation.py --output_dir $SGD_path --optimizer 'SGD' --init_lr $lr --reg_param $rp --n_iters 5000 --param_file 'params_dump_SGD.txt' --neuron $neuron_idx --n_out_neuron 2 --layer 'fc9'
			fi

		done

	done
done
