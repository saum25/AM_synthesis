#!/bin/bash

# best and simple way for multiline comment
# : ' in the start and end. Watch out for any ' in between.'

clear all

# remove all the results
rm -rf results

# if needed run the code for SGD optimisation
# 1 -> run for SGD as well
# 0 -> only Adam

sgd_optm=0
act_min=0

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
iter_str="_iter_"
start_str="seed_"

# watch out there is no space between '=' sign
#scale_factor=0.1

# variable to assist in writing header in the csv only for the first setting of hyper parameters
setting_count=1

echo --------- optimisation starts ---------------
echo 

# number of optimisation steps
for iter in 100 500 1000
do
	for lr in 0.1 0.01 0.001
	do
		# bash does''t support floating point arithmetic, so use bc to process
		# watch for the use of $ sign while saving the output to another variable
		# currently not used
		#adaptive_reg=$(echo $lr*$scale_factor | bc -l)

		for rp in 0.1 0.01 0.001
		do
			echo ++++++++++++++++++++ learning rate: $lr regularisation paramter: $rp iterations: $iter hyperparam setting count: $setting_count ++++++++++++++
			mkdir results/Adam/maximise/$lr_str$lr$rp_str$rp$iter_str$iter
			if [ $act_min -eq 1 ]; then
				mkdir results/Adam/minimise/$lr_str$lr$rp_str$rp$iter_str$iter
			fi

			for start in {0..49}
			do
				echo ++++++++++++++++++++ seed $start ++++++++++++++

				echo 
				echo +++++++ Adam - activation maximisation case ++++++++
				mkdir results/Adam/maximise/$lr_str$lr$rp_str$rp$iter_str$iter/$start_str$start
				mkdir results/Adam/maximise/$lr_str$lr$rp_str$rp$iter_str$iter/$start_str$start/examples
				python activation_maximisation.py --output_dir results/Adam/maximise --init_lr $lr --reg_param $rp --n_iters $iter --seed $start --stats_csv 'results/optm_stats_Adam.csv' --count $setting_count --n_out_neurons 2 --neuron 0

				if [ $sgd_optm -eq 1 ]; then
					echo +++++++ SGD - activation maximisation case ++++++++
					mkdir results/SGD/maximise/$lr_str$lr$rp_str$rp
					mkdir results/SGD/maximise/$lr_str$lr$rp_str$rp/examples
					python activation_maximisation.py --output_dir results/SGD/maximise --optimizer 'SGD' --init_lr $lr --reg_param $rp --n_iters $iter --seed $start --stats_csv 'results/optm_stats_SGD.csv' --count $setting_count
				fi


				if [ $act_min -eq 1 ];  then
					echo +++++++ Adam - activation minimisation case ++++++++
					mkdir results/Adam/minimise/$lr_str$lr$rp_str$rp$iter_str$iter/$start_str$start
					mkdir results/Adam/minimise/$lr_str$lr$rp_str$rp$iter_str$iter/$start_str$start/examples
					python activation_maximisation.py --output_dir results/Adam/minimise --init_lr $lr --reg_param $rp --n_iters $iter --seed $start --stats_csv 'results/optm_stats_Adam.csv' --count $setting_count --minimise 
				fi

				setting_count=$((setting_count+1))


				if [[ $sgd_optm -eq 1 && $act_min -eq 1 ]]; then
					echo +++++++ SGD - activation minimisation case ++++++++
					mkdir results/SGD/minimise/$lr_str$lr$rp_str$rp
					mkdir results/SGD/minimise/$lr_str$lr$rp_str$rp/examples
					python activation_maximisation.py --output_dir results/SGD/minimise --optimizer 'SGD' --init_lr $lr --reg_param $rp --n_iters $iter --stats_csv 'results/optm_stats_SGD.csv' --count $setting_count --minimise
				fi
			done

		done

	done
done
echo ----------optimisation ends -------------------