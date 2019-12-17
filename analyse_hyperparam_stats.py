'''
Created on 27 Feb 2019

@author: Saumitra
'''
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import FID
import seaborn as sns

N_excerpts=50
N_params = 27
am_act=[]

# load csv
# depending on the case, we need to manually set the csv file to load the activations from
#input_file='results_iclr_settings_minimise_right_objective/results/optm_stats_Adam.csv'
#input_file = 'results_iclr_maximise/results/optm_stats_Adam.csv'
input_file = 'results_two_neuron_model_thesis_configs/results_two_neuron_model_neuron_idx_0/optm_stats_Adam.csv'
optm_stats=pd.read_csv(input_file)
acts=optm_stats[['Max/Min Activation']].values
print "shape acts column from csv"
print acts.shape
print type(acts)

# read the learning rate, regularisation param and iterations information as well
lr = optm_stats[['Learning Rate']].values
rp = optm_stats[['Regularisation Param']].values
iterations = optm_stats[['Iterations']].values

# cluster the activations, where each cluster corresponds to a parameter setting
for cluster, idx in enumerate(np.arange(0, N_params * N_excerpts, N_excerpts)):
    am_act.append((cluster, (lr[idx][0], rp[idx][0], iterations[idx][0]), acts[idx:idx+N_excerpts, 0])) # list of 1-d arrays

# calculate mean and variances (unbiased) for each parameter setting    
#print am_act
m2_list=[]
s2_list=[]
for ele in am_act:
    m2_list.append((ele[0], ele[1], np.mean(ele[2])))
    s2_list.append((ele[0], ele[1], np.var(ele[2], ddof=1)))

print m2_list
print s2_list

'''am_act_array = np.vstack(am_act)
print "shape of am act array",
print am_act_array.shape
print am_act_array[0]'''

# load .npz from dataset
#file_path = 'dataset_analysis/one_neuron_model_sigmoid_split/fc9_analysis.npz'
file_path = 'dataset_analysis/two_neuron_model_model2/neuron_0/fc9_analysis.npz'
with np.load(file_path) as fp:
        # list of np arrays
        ana_data = [fp[ele] for ele in sorted(fp.files)]

# selects the max values: need to modify for the min values
dataset_act_array = ana_data[1] # max case
#dataset_act_array = ana_data[4] # min case
print "shape of dataset act array",
print dataset_act_array.shape
print dataset_act_array

# from dataset
m1 = np.mean(dataset_act_array)
print("mean activation over dataset examples: %f" %m1)
s1=np.var(dataset_act_array, ddof=1)
print("variance over dataset examples: %f" %s1)

# from AM
'''m2 = np.mean(am_act_array, axis=1)
print("mean activation shape over AM examples:"),
print(m2.shape)
print m2

s2=np.var(am_act_array, axis=1)
print("variance shape over AM examples:"),
print(s2.shape)
print(s2)'''

distances = []

for m2, s2 in zip(m2_list, s2_list):
    #distances.append((m2[0], FID.calculate_frechet_distance(mu1=m1, sigma1=s1, mu2=m2[1], sigma2=s2[1])))
    distances.append((m2[0], m2[1], FID.calculate_frechet_distance(mu1=m2[2], sigma1=s2[2], mu2=m1, sigma2=s1)))

print
print "unsorted (all informations):",
print(distances)
print 

indices=[]
dist=[]

for ele in distances:
    indices.append(ele[0])
    dist.append(ele[2])
    
N = 27
max_f=False

top_N_act = sorted(dist, reverse = max_f)[:N]
top_N_act_ids = sorted(range(len(dist)), key = lambda i: dist[i], reverse = max_f)[:N]
top_N_mel = [distances[idx] for idx in top_N_act_ids]

print "ascending sorting (only distance):",
print top_N_act
print

print "ascending sorting (only indexes):",
print top_N_act_ids
print

print "ascending sorting (all informations):"
for ele in top_N_mel:
    print ele
    
'''# list of dicts -> pandas dataframe
temp = np.zeros((len(top_N_act), 2))
for idx, ele in enumerate(top_N_act):
    temp[idx][0] = ele
    temp[idx][1] = 1
print temp.shape

df_acts = pd.DataFrame(temp, columns=['distances', 'labels'])
df_acts.to_csv('hyper-param_stats.csv')

# plotting the distribution of neurons
sns.set(color_codes=True)
sns.boxplot(y='distances', x='labels', data=df_acts)
plt.savefig('dist.pdf', dpi=300)'''



