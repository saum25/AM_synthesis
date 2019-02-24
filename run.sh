#!/bin/bash

# best and simple way for multiline comment
# : ' in the start and end. Watch out for any ' in between.'

clear all

# remove all the results
rm -rf results

# if needed run the code for SGD optimisation
# 1 -> run for SGD as well
# 0 -> only Adam

sgd_optm=1
act_min=1

# create results directory
mkdir results

mkdir results/Adam
mkdir results/Adam/maximise
if [ $act_min -eq 1 ]; then
	mkdir results/Adam/minimise
fi


if [ $sgd_optm -eq 1 ]; then
	mkdir results/SGD
	mkdir results/SGD/maximise
	if [ $act_min -eq 1 ]; then
		mkdir results/SGD/minimise
	fi
fi

# string variables for naming the results directory
lr_str="lr_"
rp_str="_rp_"

# watch out there is no space between '=' sign
#scale_factor=0.1

# variable to assist in writing header in the csv only for the first setting of hyper parameters
setting_count=1

# number of optimisation steps
iter=5000

echo --------- optimisation starts ---------------
echo 

for lr in 1.0 0.1 0.01 0.001 0.0001
do
	# bash does''t support floating point arithmetic, so use bc to process
	# watch for the use of $ sign while saving the output to another variable
	# currently not used
	#adaptive_reg=$(echo $lr*$scale_factor | bc -l)

	for rp in 1.0 0.1 0.01 0.001 0.0001
	do
		echo ++++++++++++++++++++ learning rate: $lr regularisation paramter: $rp hyperparam setting count: $setting_count ++++++++++++++

		echo 
		echo +++++++ Adam - activation maximisation case ++++++++
		mkdir results/Adam/maximise/$lr_str$lr$rp_str$rp
		mkdir results/Adam/maximise/$lr_str$lr$rp_str$rp/examples
		python activation_maximisation.py --output_dir results/Adam/maximise --init_lr $lr --reg_param $rp --n_iters $iter --stats_csv 'results/optm_stats_Adam.csv' --count $setting_count

		if [ $sgd_optm -eq 1 ]; then
			echo +++++++ SGD - activation maximisation case ++++++++
			mkdir results/SGD/maximise/$lr_str$lr$rp_str$rp
			mkdir results/SGD/maximise/$lr_str$lr$rp_str$rp/examples
			python activation_maximisation.py --output_dir results/SGD/maximise --optimizer 'SGD' --init_lr $lr --reg_param $rp --n_iters $iter --stats_csv 'results/optm_stats_SGD.csv' --count $setting_count
		fi

		setting_count=$((setting_count+1))


		if [ $act_min -eq 1 ];  then
			echo +++++++ Adam - activation minimisation case ++++++++
			mkdir results/Adam/minimise/$lr_str$lr$rp_str$rp
			mkdir results/Adam/minimise/$lr_str$lr$rp_str$rp/examples
			python activation_maximisation.py --output_dir results/Adam/minimise --init_lr $lr --reg_param $rp --n_iters $iter --stats_csv 'results/optm_stats_Adam.csv' --count $setting_count --minimise 
		fi


		if [[ $sgd_optm -eq 1 && $act_min -eq 1 ]]; then
			echo +++++++ SGD - activation minimisation case ++++++++
			mkdir results/SGD/minimise/$lr_str$lr$rp_str$rp
			mkdir results/SGD/minimise/$lr_str$lr$rp_str$rp/examples
			python activation_maximisation.py --output_dir results/SGD/minimise --optimizer 'SGD' --init_lr $lr --reg_param $rp --n_iters $iter --stats_csv 'results/optm_stats_SGD.csv' --count $setting_count --minimise
		fi


	done

done

echo ----------optimisation ends -------------------