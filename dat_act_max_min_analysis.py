'''
Created on 29 Jan 2019

Uses the available .npz file
1. read the saved arrays of mel spectrograms, activations and predictions for a neuron
2. saves plots of all mel spectrograms
3. inverts mel spects to time domain
@author: Saumitra
'''

import numpy as np
import Utils

file_path = 'dataset_analysis/act_out_fc8_tr_n3_analysis.npz'
results_path = 'results/from_dataset'
mean_std_fp = 'models/classifier/jamendo_meanstd.npz'
N_start = 0 # start index of excerpts to analyse
N_stop = 10 # stop index of excerpts to analyse

try:
    with np.load(file_path) as fp:
        # list of np arrays
        ana_data = [fp[ele][N_start:N_stop] for ele in sorted(fp.files)]
        
        # load training data set-wise mean and std dev
        with np.load(mean_std_fp) as f:
            mean = f['mean']
            std = f['std']
        
        # loop over max and min cases
        # sequence is max examples case followed by min examples case
        for idx in range(0, 4, 3):                
            # select case
            if idx == 0:
                updated_rp = results_path+'/'+ 'max_case'
                var = 'max'
            elif idx == 3:
                updated_rp = results_path+'/'+ 'min_case'
                var='min'
            else:
                raise ValueError("range index should be 0 or 3!")

            print "=====looping over [%s] case=======" %var
            
            for mel, act, pred in zip(ana_data[idx], ana_data[idx+1], ana_data[idx+2]):
                
                print("activation: %f prediction:%f" %(act, pred))
                
                # unnormalise the mel spectrograms
                mel = (mel*std)+mean
                
                # save mel spect plots
                Utils.save_mel(mel.T, directory=updated_rp, score=act, pred=pred, case='dataset')
                # save audio after mel inversion
                Utils.save_audio(mel.T, updated_rp, act, pred, hopsize=315)
                
except IOError as error:
    print error