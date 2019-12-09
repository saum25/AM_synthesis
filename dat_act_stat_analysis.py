'''
Created on 12 Dec 2018

Script to analyse activations of a neuron. We generate these activations
by passing N excerpts to the SVD model and recording the output of the
neuron for each case. Activations are stored in a dictionary with file name 
as key and all activations for that file as the value. On execution this
script saves two outputs

1) a CSV file with rows corresponding to number of neurons, columns to the analysis
stats for each. We save max act, min act, mean act, median act and std dev of act for
eah neuron.

2) a boxplot to analyse the distribution of acts and outliers for each neuron.

@author: Saumitra
'''

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import OrderedDict

results_path= 'results/from_dataset/stat_analysis'
act_files = ['acts_tr_neuron 0']
n_cols = 2
n_excerpts = 0
iteration=0
all_activations=[]
neuron_stats=[]

# create additional lists
neurons = [f.split('_')[2] for f in act_files]
neuron_idx = [int(c) for n in neurons for c in n.split() if c.isdigit()]

try:    
    for neuron, act_file in zip(neurons, act_files):
        print "======= analysing %s =======" %neuron
        
        activation_file_path = 'dataset_analysis'+ '/'+'one_neuron_model_sigmoid_split/' + act_file + '.npz'
        print "Reading activations from: %s" %activation_file_path
    
        # load activations for the selected neuron
        act_list = []
        
        with np.load(activation_file_path) as activations:
            act_dict = {fn:activations[fn] for fn in activations.files}
            for key, value in act_dict.items():
                # list of 1-d activation arrays
                if value.ndim==1:
                    act_list.append(value)
                else:
                    act_list.append(value.ravel())
        # column-wise stacking
        act_array = np.hstack(act_list)
        print("[Shape] stacked activation array:"),
        print(act_array.shape)
        print

        # a list of 1-d numpy arrays
        all_activations.append(act_array)

        # save key stats ( as an ordered dictionary) for each neuron
        neuron_stats.append(OrderedDict([('max', np.amax(act_array)), ('min', np.amin(act_array)), ('mean', np.mean(act_array)), ('median', np.median(act_array)), ('std dev', np.std(act_array))]))

    # list of dicts -> pandas dataframe
    df_stats = pd.DataFrame(neuron_stats, index=neurons)
    df_stats.to_csv(results_path+ '/'+'neuron_stats.csv')

    # concatenate all the activation arrays column-wise
    for ele in all_activations:
        n_excerpts += len(ele)
    data = np.zeros((n_excerpts, n_cols))

    for neuron_id, activations in zip(neuron_idx, all_activations):
        data[iteration:iteration+len(activations), 0] = neuron_id
        data[iteration:iteration+len(activations), 1] = activations
        iteration += len(activations)
    print("[Shape] concatenated all activations array"),
    print(data.shape)   
    
    # create a pandas data frame as seaborn expects one
    df_acts = pd.DataFrame(data, columns=['neuron_idx', 'activations'])
    df_acts.neuron_idx = df_acts['neuron_idx'].astype('int') # change the dtype
        
    # plotting the distribution of neurons
    sns.set(color_codes=True)
    sns.boxplot(x='activations', y='neuron_idx', data=df_acts, width=0.5, orient='h')
    plt.savefig(results_path+'/'+ 'dist.pdf', dpi=300)
               
except IOError as error:
    print(error)
