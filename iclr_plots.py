'''
Created on 19 Apr 2019

@author: Saumitra
'''

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import librosa.display as disp
from matplotlib import rc

def normalise(x):
    """
    Normalise an input vector/ matrix in the range 0 - 1
    @param: x: input vector/matrix
    @return: normalised vector/matrix
    """
    if x.max() == x.min(): # wierd case in the SGD optimisation, where in an intermediate step this happens
        return x
    else:
        return((x-x.min())/(x.max()-x.min()))
    
def fig3():

    file_path1 ='iclr_paper_results/Results1-hyperparam-search/for_camera_ready_plots/iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_4.npz'
    file_path2 ='iclr_paper_results/Results1-hyperparam-search/for_camera_ready_plots/iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_10.npz'
    file_path3 ='iclr_paper_results/Results1-hyperparam-search/for_camera_ready_plots/iclr_inp_mel_best_mel_lr_0.01_rp_0.01_iter_500_seed_4.npz'
    file_path4 ='iclr_paper_results/Results1-hyperparam-search/for_camera_ready_plots/iclr_inp_mel_best_mel_lr_0.01_rp_0.01_iter_500_seed_10.npz'
    file_path5 ='iclr_paper_results/Results1-hyperparam-search/for_camera_ready_plots/iclr_inp_mel_best_mel_lr_0.1_rp_0.001_iter_1000_seed_4.npz'
    file_path6 ='iclr_paper_results/Results1-hyperparam-search/for_camera_ready_plots/iclr_inp_mel_best_mel_lr_0.1_rp_0.001_iter_1000_seed_10.npz'
    
    with np.load(file_path1) as fp:
            # list of np arrays
            ana_data1 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path2) as fp:
            # list of np arrays
            ana_data2 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path3) as fp:
            # list of np arrays
            ana_data3 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path4) as fp:
            # list of np arrays
            ana_data4 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path5) as fp:
            # list of np arrays
            ana_data5 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path6) as fp:
            # list of np arrays
            ana_data6 = [fp[ele] for ele in sorted(fp.files)]
            
    
    plt.figure(figsize=(10, 4)) 
    
    plt.subplot(2, 4, 1)
    disp.specshow(normalise(ana_data1[0]), x_axis=None, hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.ylabel('Freq(Hz)')
    plt.title('Input')
    
    plt.subplot(2, 4, 2)
    disp.specshow(normalise(ana_data1[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.title(r'$C_1$')
    
    plt.subplot(2, 4, 3)
    disp.specshow(normalise(ana_data3[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.title(r'$C_2$')
    
    plt.subplot(2, 4, 4)
    disp.specshow(normalise(ana_data5[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.title(r'$C_3$')
    
    plt.subplot(2, 4, 5)
    disp.specshow(normalise(ana_data2[0]), x_axis='time', hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    plt.ylabel('Freq(Hz)')
    
    plt.subplot(2, 4, 6)
    disp.specshow(normalise(ana_data2[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    
    plt.subplot(2, 4, 7)
    disp.specshow(normalise(ana_data4[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    
    plt.subplot(2, 4, 8)
    disp.specshow(normalise(ana_data6[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    cax = plt.axes([0.91, 0.11, 0.0125, 0.77])
    cbar = plt.colorbar(cax=cax)
    plt.savefig('iclr_paper_results/Results1-hyperparam-search/for_camera_ready_plots/fig3_camera_ready.pdf')
    
# CR fig4
def fig4():
    # activate latex text rendering
    #rc('text', usetex=True)
    data_fp = 'iclr_paper_results/Results2-vocal-non-vocal/for_camera_ready_plots/'
    file_path1 = data_fp+'iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_12_minimise_False.npz'
    file_path2 = data_fp + 'iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_12_minimise_True.npz'
    file_path3 = data_fp + 'iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_42_minimise_False.npz'
    file_path4 = data_fp + 'iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_42_minimise_True.npz'
    file_path5 = data_fp + 'iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_20_minimise_False.npz'
    file_path6 = data_fp + 'iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_20_minimise_True.npz'
    file_path7 = data_fp + 'iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_11_minimise_False.npz'
    file_path8 = data_fp + 'iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_11_minimise_True.npz'
    
    with np.load(file_path1) as fp:
            # list of np arrays
            ana_data1 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path2) as fp:
            # list of np arrays
            ana_data2 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path3) as fp:
            # list of np arrays
            ana_data3 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path4) as fp:
            # list of np arrays
            ana_data4 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path5) as fp:
            # list of np arrays
            ana_data5 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path6) as fp:
            # list of np arrays
            ana_data6 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path7) as fp:
            # list of np arrays
            ana_data7 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path8) as fp:
            # list of np arrays
            ana_data8 = [fp[ele] for ele in sorted(fp.files)]
            
    
    plt.figure(figsize=(12, 6)) 
    
    plt.subplot(3, 4, 1)
    disp.specshow(normalise(ana_data1[0]), x_axis=None, hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.ylabel('$\\bf{Input}$ \n Freq(Hz)', multialignment='center')
    #plt.title('Input')
    
    plt.subplot(3, 4, 2)
    disp.specshow(normalise(ana_data3[0]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    #plt.title(r'$C_1$')
    
    plt.subplot(3, 4, 3)
    disp.specshow(normalise(ana_data5[0]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    #plt.title(r'$C_2$')
    
    plt.subplot(3, 4, 4)
    disp.specshow(normalise(ana_data7[0]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    #plt.title(r'$C_3$')
    
    plt.subplot(3, 4, 5)
    disp.specshow(normalise(ana_data1[1]), x_axis=None, hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    #plt.xlabel('Time(sec)')
    plt.ylabel('$ \\bf{Maximisation}$ \n Freq(Hz)', multialignment='center')
    
    plt.subplot(3, 4, 6)
    disp.specshow(normalise(ana_data3[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    #plt.xlabel('Time(sec)')
    
    plt.subplot(3, 4, 7)
    disp.specshow(normalise(ana_data5[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    #plt.xlabel('Time(sec)')
    
    plt.subplot(3, 4, 8)
    disp.specshow(normalise(ana_data7[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    #plt.xlabel('Time(sec)')
    
    plt.subplot(3, 4, 9)
    disp.specshow(normalise(ana_data2[1]), x_axis='time', hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    plt.ylabel('$\\bf{Minimisation}$ \n Freq(Hz)', multialignment='center')
    
    plt.subplot(3, 4, 10)
    disp.specshow(normalise(ana_data4[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    
    plt.subplot(3, 4, 11)
    disp.specshow(normalise(ana_data6[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    
    plt.subplot(3, 4, 12)
    disp.specshow(normalise(ana_data8[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    
    plt.subplots_adjust(wspace=0.05, hspace=0.06)
    cax = plt.axes([0.91, 0.11, 0.0125, 0.77])
    cbar = plt.colorbar(cax=cax)
    plt.savefig(data_fp+'fig4_camera_ready.pdf')

# minimisation case - hyper_param_search
def thesis_fig_5_3():

    # C1 - Best
    file_path1 ='thesis_plots/hyper_param_search_onm/minimise/data_for_plots/iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_1000_seed_7_minimise_True.npz'
    file_path2 ='thesis_plots/hyper_param_search_onm/minimise/data_for_plots/iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_1000_seed_12_minimise_True.npz'
    file_path3 ='thesis_plots/hyper_param_search_onm/minimise/data_for_plots/iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_1000_seed_30_minimise_True.npz'
    file_path4 ='thesis_plots/hyper_param_search_onm/minimise/data_for_plots/iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_1000_seed_49_minimise_True.npz'
    # C2 - Median
    file_path5 ='thesis_plots/hyper_param_search_onm/minimise/data_for_plots/iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_100_seed_7_minimise_True.npz'
    file_path6 ='thesis_plots/hyper_param_search_onm/minimise/data_for_plots/iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_100_seed_12_minimise_True.npz'
    file_path7 ='thesis_plots/hyper_param_search_onm/minimise/data_for_plots/iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_100_seed_30_minimise_True.npz'
    file_path8 ='thesis_plots/hyper_param_search_onm/minimise/data_for_plots/iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_100_seed_49_minimise_True.npz'
    # C3 - Worst
    file_path9 ='thesis_plots/hyper_param_search_onm/minimise/data_for_plots/iclr_inp_mel_best_mel_lr_0.1_rp_0.001_iter_1000_seed_7_minimise_True.npz'
    file_path10 ='thesis_plots/hyper_param_search_onm/minimise/data_for_plots/iclr_inp_mel_best_mel_lr_0.1_rp_0.001_iter_1000_seed_12_minimise_True.npz'
    file_path11 ='thesis_plots/hyper_param_search_onm/minimise/data_for_plots/iclr_inp_mel_best_mel_lr_0.1_rp_0.001_iter_1000_seed_30_minimise_True.npz'
    file_path12 ='thesis_plots/hyper_param_search_onm/minimise/data_for_plots/iclr_inp_mel_best_mel_lr_0.1_rp_0.001_iter_1000_seed_49_minimise_True.npz'    
    
    with np.load(file_path1) as fp:
            # list of np arrays
            ana_data1 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path2) as fp:
            # list of np arrays
            ana_data2 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path3) as fp:
            # list of np arrays
            ana_data3 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path4) as fp:
            # list of np arrays
            ana_data4 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path5) as fp:
            # list of np arrays
            ana_data5 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path6) as fp:
            # list of np arrays
            ana_data6 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path7) as fp:
            # list of np arrays
            ana_data7 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path8) as fp:
            # list of np arrays
            ana_data8 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path9) as fp:
            # list of np arrays
            ana_data9 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path10) as fp:
            # list of np arrays
            ana_data10 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path11) as fp:
            # list of np arrays
            ana_data11 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path12) as fp:
            # list of np arrays
            ana_data12 = [fp[ele] for ele in sorted(fp.files)]
                        
    
    plt.figure(figsize=(12, 8)) 
    
    plt.subplot(4, 4, 1)
    disp.specshow(normalise(ana_data1[0]), x_axis=None, hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.ylabel('$ \\bf{Seed = 7}$ \n Freq(Hz)', multialignment='center')
    plt.title(r'$f_g(\tilde{z}_i)$')
    
    plt.subplot(4, 4, 2)
    disp.specshow(normalise(ana_data1[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.title(r'$C^{min}_1$')
    
    plt.subplot(4, 4, 3)
    disp.specshow(normalise(ana_data5[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.title(r'$C^{min}_2$')
    
    plt.subplot(4, 4, 4)
    disp.specshow(normalise(ana_data9[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.title(r'$C^{min}_3$')
    
    plt.subplot(4, 4, 5)
    disp.specshow(normalise(ana_data2[0]), x_axis=None, hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    #plt.xlabel('Time(sec)')
    plt.ylabel('$ \\bf{Seed = 12}$ \n Freq(Hz)', multialignment='center')
    
    plt.subplot(4, 4, 6)
    disp.specshow(normalise(ana_data2[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    #plt.xlabel('Time(sec)')
    
    plt.subplot(4, 4, 7)
    disp.specshow(normalise(ana_data6[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    #plt.xlabel('Time(sec)')
    
    plt.subplot(4, 4, 8)
    disp.specshow(normalise(ana_data10[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    #plt.xlabel('Time(sec)')
    
    plt.subplot(4, 4, 9)
    disp.specshow(normalise(ana_data3[0]), x_axis=None, hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    #plt.xlabel('Time(sec)')
    plt.ylabel('$ \\bf{Seed = 30}$ \n Freq(Hz)', multialignment='center')
    
    plt.subplot(4, 4, 10)
    disp.specshow(normalise(ana_data3[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    #plt.xlabel('Time(sec)')
    
    plt.subplot(4, 4, 11)
    disp.specshow(normalise(ana_data7[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    #plt.xlabel('Time(sec)')
    
    plt.subplot(4, 4, 12)
    disp.specshow(normalise(ana_data11[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    #plt.xlabel('Time(sec)')

    plt.subplot(4, 4, 13)
    disp.specshow(normalise(ana_data4[0]), x_axis='time', hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    plt.ylabel('$ \\bf{Seed = 49}$ \n Freq(Hz)', multialignment='center')
    
    plt.subplot(4, 4, 14)
    disp.specshow(normalise(ana_data4[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    
    plt.subplot(4, 4, 15)
    disp.specshow(normalise(ana_data8[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    
    plt.subplot(4, 4, 16)
    disp.specshow(normalise(ana_data12[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')

    
    plt.subplots_adjust(wspace=0.08, hspace=0.1)
    # plots the colorbar at the required position
    cax = plt.axes([0.92, 0.105, 0.012, 0.78])
    cbar = plt.colorbar(cax=cax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.savefig('thesis_plots/hyper_param_search_onm/minimise/data_for_plots/am_hyper_param_search_min.pdf', bbox='tight', dpi=300)

# maximisation case - hyper_param_search
def thesis_fig_5_4():

    # C1 - Best
    file_path1 ='thesis_plots/hyper_param_search_onm/maximise/data_for_plots/iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_4_minimise_False.npz'
    file_path2 ='thesis_plots/hyper_param_search_onm/maximise/data_for_plots/iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_10_minimise_False.npz'
    file_path3 ='thesis_plots/hyper_param_search_onm/maximise/data_for_plots/iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_21_minimise_False.npz'
    file_path4 ='thesis_plots/hyper_param_search_onm/maximise/data_for_plots/iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_39_minimise_False.npz'
    # C2 - Median
    file_path5 ='thesis_plots/hyper_param_search_onm/maximise/data_for_plots/iclr_inp_mel_best_mel_lr_0.01_rp_0.1_iter_1000_seed_4_minimise_False.npz'
    file_path6 ='thesis_plots/hyper_param_search_onm/maximise/data_for_plots/iclr_inp_mel_best_mel_lr_0.01_rp_0.1_iter_1000_seed_10_minimise_False.npz'
    file_path7 ='thesis_plots/hyper_param_search_onm/maximise/data_for_plots/iclr_inp_mel_best_mel_lr_0.01_rp_0.1_iter_1000_seed_21_minimise_False.npz'
    file_path8 ='thesis_plots/hyper_param_search_onm/maximise/data_for_plots/iclr_inp_mel_best_mel_lr_0.01_rp_0.1_iter_1000_seed_39_minimise_False.npz'
    # C3 - Worst
    file_path9 ='thesis_plots/hyper_param_search_onm/maximise/data_for_plots/iclr_inp_mel_best_mel_lr_0.1_rp_0.001_iter_1000_seed_4_minimise_False.npz'
    file_path10 ='thesis_plots/hyper_param_search_onm/maximise/data_for_plots/iclr_inp_mel_best_mel_lr_0.1_rp_0.001_iter_1000_seed_10_minimise_False.npz'
    file_path11 ='thesis_plots/hyper_param_search_onm/maximise/data_for_plots/iclr_inp_mel_best_mel_lr_0.1_rp_0.001_iter_1000_seed_21_minimise_False.npz'
    file_path12 ='thesis_plots/hyper_param_search_onm/maximise/data_for_plots/iclr_inp_mel_best_mel_lr_0.1_rp_0.001_iter_1000_seed_39_minimise_False.npz'    
    
    with np.load(file_path1) as fp:
            # list of np arrays
            ana_data1 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path2) as fp:
            # list of np arrays
            ana_data2 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path3) as fp:
            # list of np arrays
            ana_data3 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path4) as fp:
            # list of np arrays
            ana_data4 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path5) as fp:
            # list of np arrays
            ana_data5 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path6) as fp:
            # list of np arrays
            ana_data6 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path7) as fp:
            # list of np arrays
            ana_data7 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path8) as fp:
            # list of np arrays
            ana_data8 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path9) as fp:
            # list of np arrays
            ana_data9 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path10) as fp:
            # list of np arrays
            ana_data10 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path11) as fp:
            # list of np arrays
            ana_data11 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path12) as fp:
            # list of np arrays
            ana_data12 = [fp[ele] for ele in sorted(fp.files)]
                        
    
    plt.figure(figsize=(12, 8)) 
    
    plt.subplot(4, 4, 1)
    disp.specshow(normalise(ana_data1[0]), x_axis=None, hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.ylabel('$ \\bf{Seed = 4}$ \n Freq(Hz)', multialignment='center')
    plt.title(r'$f_g(\tilde{z}_i)$')
    
    plt.subplot(4, 4, 2)
    disp.specshow(normalise(ana_data1[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.title(r'$C^{max}_1$')
    
    plt.subplot(4, 4, 3)
    disp.specshow(normalise(ana_data5[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.title(r'$C^{max}_2$')
    
    plt.subplot(4, 4, 4)
    disp.specshow(normalise(ana_data9[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.title(r'$C^{max}_3$')
    
    plt.subplot(4, 4, 5)
    disp.specshow(normalise(ana_data2[0]), x_axis=None, hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    #plt.xlabel('Time(sec)')
    plt.ylabel('$ \\bf{Seed = 10}$ \n Freq(Hz)', multialignment='center')
    
    plt.subplot(4, 4, 6)
    disp.specshow(normalise(ana_data2[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    #plt.xlabel('Time(sec)')
    
    plt.subplot(4, 4, 7)
    disp.specshow(normalise(ana_data6[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    #plt.xlabel('Time(sec)')
    
    plt.subplot(4, 4, 8)
    disp.specshow(normalise(ana_data10[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    #plt.xlabel('Time(sec)')
    
    plt.subplot(4, 4, 9)
    disp.specshow(normalise(ana_data3[0]), x_axis=None, hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    #plt.xlabel('Time(sec)')
    plt.ylabel('$ \\bf{Seed = 21}$ \n Freq(Hz)', multialignment='center')
    
    plt.subplot(4, 4, 10)
    disp.specshow(normalise(ana_data3[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    #plt.xlabel('Time(sec)')
    
    plt.subplot(4, 4, 11)
    disp.specshow(normalise(ana_data7[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    #plt.xlabel('Time(sec)')
    
    plt.subplot(4, 4, 12)
    disp.specshow(normalise(ana_data11[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    #plt.xlabel('Time(sec)')

    plt.subplot(4, 4, 13)
    disp.specshow(normalise(ana_data4[0]), x_axis='time', hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    plt.ylabel('$ \\bf{Seed = 39}$ \n Freq(Hz)', multialignment='center')
    
    plt.subplot(4, 4, 14)
    disp.specshow(normalise(ana_data4[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    
    plt.subplot(4, 4, 15)
    disp.specshow(normalise(ana_data8[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    
    plt.subplot(4, 4, 16)
    disp.specshow(normalise(ana_data12[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')

    
    plt.subplots_adjust(wspace=0.08, hspace=0.1)
    # plots the colorbar at the required position
    cax = plt.axes([0.92, 0.105, 0.012, 0.78])
    cbar = plt.colorbar(cax=cax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.savefig('thesis_plots/hyper_param_search_onm/maximise/data_for_plots/am_hyper_param_search_max.pdf', bbox='tight', dpi=300)

# maximisation and minisation from the same seed with best settings for each one case
def thesis_fig_5_5():
    data_fp = 'thesis_plots/max_min_onm/data_for_plots/'
    # max - best
    file_path1 = data_fp + 'iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_2_minimise_False.npz'
    file_path2 = data_fp + 'iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_14_minimise_False.npz'
    file_path3 = data_fp + 'iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_26_minimise_False.npz'
    file_path4 = data_fp + 'iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_44_minimise_False.npz'
    file_path5 = data_fp + 'iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_47_minimise_False.npz'
    # max - median
    file_path6 = data_fp + 'iclr_inp_mel_best_mel_lr_0.01_rp_0.1_iter_1000_seed_2_minimise_False.npz'
    file_path7 = data_fp + 'iclr_inp_mel_best_mel_lr_0.01_rp_0.1_iter_1000_seed_14_minimise_False.npz'
    file_path8 = data_fp + 'iclr_inp_mel_best_mel_lr_0.01_rp_0.1_iter_1000_seed_26_minimise_False.npz'
    file_path9 = data_fp + 'iclr_inp_mel_best_mel_lr_0.01_rp_0.1_iter_1000_seed_44_minimise_False.npz'
    file_path10 = data_fp + 'iclr_inp_mel_best_mel_lr_0.01_rp_0.1_iter_1000_seed_47_minimise_False.npz'
    # min - best
    file_path11 = data_fp + 'iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_1000_seed_2_minimise_True.npz'
    file_path12 = data_fp + 'iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_1000_seed_14_minimise_True.npz'
    file_path13 = data_fp + 'iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_1000_seed_26_minimise_True.npz'
    file_path14 = data_fp + 'iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_1000_seed_44_minimise_True.npz'
    file_path15 = data_fp + 'iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_1000_seed_47_minimise_True.npz'
    
    with np.load(file_path1) as fp:
            # list of np arrays
            ana_data1 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path2) as fp:
            # list of np arrays
            ana_data2 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path3) as fp:
            # list of np arrays
            ana_data3 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path4) as fp:
            # list of np arrays
            ana_data4 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path5) as fp:
            # list of np arrays
            ana_data5 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path6) as fp:
            # list of np arrays
            ana_data6 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path7) as fp:
            # list of np arrays
            ana_data7 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path8) as fp:
            # list of np arrays
            ana_data8 = [fp[ele] for ele in sorted(fp.files)]

    with np.load(file_path9) as fp:
            # list of np arrays
            ana_data9 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path10) as fp:
            # list of np arrays
            ana_data10 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path11) as fp:
            # list of np arrays
            ana_data11 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path12) as fp:
            # list of np arrays
            ana_data12 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path13) as fp:
            # list of np arrays
            ana_data13 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path14) as fp:
            # list of np arrays
            ana_data14 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path15) as fp:
            # list of np arrays
            ana_data15 = [fp[ele] for ele in sorted(fp.files)]            
    
    plt.figure(figsize=(12, 6)) 
    
    # inputs
    plt.subplot(4, 5, 1)
    disp.specshow(normalise(ana_data1[0]), x_axis=None, hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.ylabel(r'$\bf{f_g(\tilde{z}_i)}$' '\n Freq(Hz)', multialignment='center')
    plt.title(r'$Seed = 2$')
    
    plt.subplot(4, 5, 2)
    disp.specshow(normalise(ana_data2[0]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.title(r'$Seed = 14$')
    
    plt.subplot(4, 5, 3)
    disp.specshow(normalise(ana_data3[0]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.title(r'$Seed = 26$')
    
    plt.subplot(4, 5, 4)
    disp.specshow(normalise(ana_data4[0]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.title(r'$Seed = 44$')
    
    plt.subplot(4, 5, 5)
    disp.specshow(normalise(ana_data5[0]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.title(r'$Seed = 47$')
    
    # max case for the best setting - lr 0.01, rp 0.001 and iter 100
    plt.subplot(4, 5, 6)
    disp.specshow(normalise(ana_data1[1]), x_axis=None, hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.ylabel('$ \\bf{Maximise^1}$ \n Freq(Hz)', multialignment='center')
    
    plt.subplot(4, 5, 7)
    disp.specshow(normalise(ana_data2[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    
    plt.subplot(4, 5, 8)
    disp.specshow(normalise(ana_data3[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    
    plt.subplot(4, 5, 9)
    disp.specshow(normalise(ana_data4[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    
    plt.subplot(4, 5, 10)
    disp.specshow(normalise(ana_data5[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')

    # max case for the median setting - lr 0.01, rp 0.1 and iter 1000
    plt.subplot(4, 5, 11)
    disp.specshow(normalise(ana_data6[1]), x_axis=None, hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.ylabel('$ \\bf{Maximise^2}$ \n Freq(Hz)', multialignment='center')
    
    plt.subplot(4, 5, 12)
    disp.specshow(normalise(ana_data7[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    
    plt.subplot(4, 5, 13)
    disp.specshow(normalise(ana_data8[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    
    plt.subplot(4, 5, 14)
    disp.specshow(normalise(ana_data9[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    
    plt.subplot(4, 5, 15)
    disp.specshow(normalise(ana_data10[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')

    # minimise case: lr 0.001, rp: 0.1 and iter 1000    
    plt.subplot(4, 5, 16)
    disp.specshow(normalise(ana_data11[1]), x_axis='time', hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    plt.ylabel('$\\bf{Minimise}$ \n Freq(Hz)', multialignment='center')
    
    plt.subplot(4, 5, 17)
    disp.specshow(normalise(ana_data12[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    
    plt.subplot(4, 5, 18)
    disp.specshow(normalise(ana_data13[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    
    plt.subplot(4, 5, 19)
    disp.specshow(normalise(ana_data14[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')

    plt.subplot(4, 5, 20)
    disp.specshow(normalise(ana_data15[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    
    plt.subplots_adjust(wspace=0.05, hspace=0.06)
    cax = plt.axes([0.91, 0.11, 0.0125, 0.77])
    cbar = plt.colorbar(cax=cax)
    plt.savefig(data_fp+'am_max_min_onm.pdf')

# maximisation case - hyper_param_search - twn
def thesis_fig_5_6():

    # C1 - Best - neuron 0
    file_path1 ='thesis_plots/hyper_param_search_tnm/neuron_0/data_for_plots/iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_1000_seed_12_minimise_False.npz'
    file_path2 ='thesis_plots/hyper_param_search_tnm/neuron_0/data_for_plots/iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_1000_seed_31_minimise_False.npz'
    
    # C2 - Median - neuron 0
    file_path3 ='thesis_plots/hyper_param_search_tnm/neuron_0/data_for_plots/iclr_inp_mel_best_mel_lr_0.01_rp_0.1_iter_1000_seed_12_minimise_False.npz'
    file_path4 ='thesis_plots/hyper_param_search_tnm/neuron_0/data_for_plots/iclr_inp_mel_best_mel_lr_0.01_rp_0.1_iter_1000_seed_31_minimise_False.npz'

    # C3 - Worst - neuron 0
    file_path5 ='thesis_plots/hyper_param_search_tnm/neuron_0/data_for_plots/iclr_inp_mel_best_mel_lr_0.1_rp_0.001_iter_1000_seed_12_minimise_False.npz'
    file_path6 ='thesis_plots/hyper_param_search_tnm/neuron_0/data_for_plots/iclr_inp_mel_best_mel_lr_0.1_rp_0.001_iter_1000_seed_31_minimise_False.npz'
    
    # C1 - Best - neuron 1
    file_path7 ='thesis_plots/hyper_param_search_tnm/neuron_1/data_for_plots/iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_10_minimise_False.npz'
    file_path8 ='thesis_plots/hyper_param_search_tnm/neuron_1/data_for_plots/iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_49_minimise_False.npz'
    
    # C2 - Median - neuron 1
    file_path9 ='thesis_plots/hyper_param_search_tnm/neuron_1/data_for_plots/iclr_inp_mel_best_mel_lr_0.01_rp_0.1_iter_1000_seed_10_minimise_False.npz'
    file_path10 ='thesis_plots/hyper_param_search_tnm/neuron_1/data_for_plots/iclr_inp_mel_best_mel_lr_0.01_rp_0.1_iter_1000_seed_49_minimise_False.npz'

    # C3 - Worst - neuron 1
    file_path11 ='thesis_plots/hyper_param_search_tnm/neuron_1/data_for_plots/iclr_inp_mel_best_mel_lr_0.1_rp_0.001_iter_1000_seed_10_minimise_False.npz'
    file_path12 ='thesis_plots/hyper_param_search_tnm/neuron_1/data_for_plots/iclr_inp_mel_best_mel_lr_0.1_rp_0.001_iter_1000_seed_49_minimise_False.npz'    
    
    with np.load(file_path1) as fp:
            # list of np arrays
            ana_data1 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path2) as fp:
            # list of np arrays
            ana_data2 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path3) as fp:
            # list of np arrays
            ana_data3 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path4) as fp:
            # list of np arrays
            ana_data4 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path5) as fp:
            # list of np arrays
            ana_data5 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path6) as fp:
            # list of np arrays
            ana_data6 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path7) as fp:
            # list of np arrays
            ana_data7 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path8) as fp:
            # list of np arrays
            ana_data8 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path9) as fp:
            # list of np arrays
            ana_data9 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path10) as fp:
            # list of np arrays
            ana_data10 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path11) as fp:
            # list of np arrays
            ana_data11 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path12) as fp:
            # list of np arrays
            ana_data12 = [fp[ele] for ele in sorted(fp.files)]
                        
    
    plt.figure(figsize=(12, 8)) 
    
    plt.subplot(4, 4, 1)
    disp.specshow(normalise(ana_data1[0]), x_axis=None, hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.ylabel('$ \\bf{Seed = 12}$ \n Freq(Hz)', multialignment='center')
    plt.title(r'$f_g(\tilde{z}_i)$')
    
    plt.subplot(4, 4, 2)
    disp.specshow(normalise(ana_data1[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.title(r'$C^{Nj}_1$')
    
    plt.subplot(4, 4, 3)
    disp.specshow(normalise(ana_data3[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.title(r'$C^{Nj}_2$')
    
    plt.subplot(4, 4, 4)
    disp.specshow(normalise(ana_data5[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.title(r'$C^{Nj}_3$')
    
    plt.subplot(4, 4, 5)
    disp.specshow(normalise(ana_data2[0]), x_axis=None, hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.ylabel('$ \\bf{Seed = 31}$ \n Freq(Hz)', multialignment='center')
    
    plt.subplot(4, 4, 6)
    disp.specshow(normalise(ana_data2[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    
    plt.subplot(4, 4, 7)
    disp.specshow(normalise(ana_data4[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    
    plt.subplot(4, 4, 8)
    disp.specshow(normalise(ana_data6[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    
    plt.subplot(4, 4, 9)
    disp.specshow(normalise(ana_data7[0]), x_axis=None, hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.ylabel('$ \\bf{Seed = 10}$ \n Freq(Hz)', multialignment='center')
    
    plt.subplot(4, 4, 10)
    disp.specshow(normalise(ana_data7[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    
    plt.subplot(4, 4, 11)
    disp.specshow(normalise(ana_data9[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    
    plt.subplot(4, 4, 12)
    disp.specshow(normalise(ana_data11[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')

    plt.subplot(4, 4, 13)
    disp.specshow(normalise(ana_data8[0]), x_axis='time', hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    plt.ylabel('$ \\bf{Seed = 49}$ \n Freq(Hz)', multialignment='center')
    
    plt.subplot(4, 4, 14)
    disp.specshow(normalise(ana_data8[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    
    plt.subplot(4, 4, 15)
    disp.specshow(normalise(ana_data10[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    
    plt.subplot(4, 4, 16)
    disp.specshow(normalise(ana_data12[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')

    
    plt.subplots_adjust(wspace=0.08, hspace=0.1)
    # plots the colorbar at the required position
    cax = plt.axes([0.92, 0.105, 0.012, 0.78])
    cbar = plt.colorbar(cax=cax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.savefig('thesis_plots/hyper_param_search_tnm/am_hyper_param_search_max_tnm.pdf', bbox='tight', dpi=300)


# qualitative comparison of the two neuron model and the one neuron model for the non-vocal content
def thesis_fig_5_7():
    data_fp = 'thesis_plots/comp_onm_tnm/data_for_plots/non-vocal/'

    # min-onm-best-config
    file_path1 = data_fp + 'onm/' +'iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_1000_seed_2_minimise_True.npz'
    file_path2 = data_fp + 'onm/' +'iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_1000_seed_11_minimise_True.npz'
    file_path3 = data_fp + 'onm/' +'iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_1000_seed_26_minimise_True.npz'
    file_path4 = data_fp + 'onm/' +'iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_1000_seed_41_minimise_True.npz'
    file_path5 = data_fp + 'onm/' +'iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_1000_seed_47_minimise_True.npz'
    
    # max-tnm-best-config-neuron-0
    file_path6 = data_fp + 'tnm_neuron_0/' +'iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_1000_seed_2_minimise_False.npz'
    file_path7 = data_fp + 'tnm_neuron_0/' +'iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_1000_seed_11_minimise_False.npz'
    file_path8 = data_fp + 'tnm_neuron_0/' +'iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_1000_seed_26_minimise_False.npz'
    file_path9 = data_fp + 'tnm_neuron_0/' +'iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_1000_seed_41_minimise_False.npz'
    file_path10 = data_fp + 'tnm_neuron_0/' +'iclr_inp_mel_best_mel_lr_0.001_rp_0.1_iter_1000_seed_47_minimise_False.npz'
        
    with np.load(file_path1) as fp:
            # list of np arrays
            ana_data1 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path2) as fp:
            # list of np arrays
            ana_data2 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path3) as fp:
            # list of np arrays
            ana_data3 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path4) as fp:
            # list of np arrays
            ana_data4 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path5) as fp:
            # list of np arrays
            ana_data5 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path6) as fp:
            # list of np arrays
            ana_data6 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path7) as fp:
            # list of np arrays
            ana_data7 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path8) as fp:
            # list of np arrays
            ana_data8 = [fp[ele] for ele in sorted(fp.files)]

    with np.load(file_path9) as fp:
            # list of np arrays
            ana_data9 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path10) as fp:
            # list of np arrays
            ana_data10 = [fp[ele] for ele in sorted(fp.files)]
    
    plt.figure(figsize=(14, 8)) 
    
    # plot inputs
    plt.subplot(3, 5, 1)
    disp.specshow(normalise(ana_data1[0]), x_axis=None, hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.ylabel(r'$\bf{f_g(\tilde{z}_i)}$' '\n Freq(Hz)', multialignment='center')
    plt.title(r'$Seed = 2$')
    
    plt.subplot(3, 5, 2)
    disp.specshow(normalise(ana_data2[0]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.title(r'$Seed = 11$')
    
    plt.subplot(3, 5, 3)
    disp.specshow(normalise(ana_data3[0]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.title(r'$Seed = 26$')
    
    plt.subplot(3, 5, 4)
    disp.specshow(normalise(ana_data4[0]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.title(r'$Seed = 41$')
    
    plt.subplot(3, 5, 5)
    disp.specshow(normalise(ana_data5[0]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.title(r'$Seed = 47$')
    
    # min-case-onm- lr 0.001, rp 0.001 and iter 1000
    plt.subplot(3, 5, 6)
    disp.specshow(normalise(ana_data1[1]), x_axis=None, hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.ylabel('$ \\bf{Minimise}$ \n Freq(Hz)', multialignment='center')
    
    plt.subplot(3, 5, 7)
    disp.specshow(normalise(ana_data2[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    
    plt.subplot(3, 5, 8)
    disp.specshow(normalise(ana_data3[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    
    plt.subplot(3, 5, 9)
    disp.specshow(normalise(ana_data4[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    
    plt.subplot(3, 5, 10)
    disp.specshow(normalise(ana_data5[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')

    # max-case-tnm-neuron idx 0 - config lr 0.001, rp 0.1 and iter 1000
    plt.subplot(3, 5, 11)
    disp.specshow(normalise(ana_data6[1]), x_axis='time', hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.ylabel('$ \\bf{Maximise}$ \n Freq(Hz)', multialignment='center')
    plt.xlabel('Time(sec)')
    
    plt.subplot(3, 5, 12)
    disp.specshow(normalise(ana_data7[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    
    plt.subplot(3, 5, 13)
    disp.specshow(normalise(ana_data8[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    
    plt.subplot(3, 5, 14)
    disp.specshow(normalise(ana_data9[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    
    plt.subplot(3, 5, 15)
    disp.specshow(normalise(ana_data10[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    
    plt.subplots_adjust(wspace=0.05, hspace=0.06)
    cax = plt.axes([0.91, 0.11, 0.0125, 0.77])
    cbar = plt.colorbar(cax=cax)
    plt.savefig(data_fp+'am_comp_onm_tnm_nv.pdf', bbox_inches='tight')
    
# qualitative comparison of the two neuron model and the one neuron model - vocal neuron case
def thesis_fig_5_8():
    
    data_fp = 'thesis_plots/comp_onm_tnm/data_for_plots/vocal/'
    
    # onm-max-best-config-lr-0.001-rp-0.001-iter-100
    file_path1 = data_fp + 'onm/' +'iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_4_minimise_False.npz'
    file_path2 = data_fp + 'onm/' +'iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_14_minimise_False.npz'
    file_path3 = data_fp + 'onm/' +'iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_16_minimise_False.npz'
    file_path4 = data_fp + 'onm/' +'iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_31_minimise_False.npz'
    file_path5 = data_fp + 'onm/' +'iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_44_minimise_False.npz'
    
    # tnm-max-neuron-idx-1best-config-lr-0.001-rp-0.001-iter-100
    file_path6 = data_fp + 'tnm_neuron_1/'+ 'iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_4_minimise_False.npz'
    file_path7 = data_fp + 'tnm_neuron_1/'+ 'iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_14_minimise_False.npz'
    file_path8 = data_fp + 'tnm_neuron_1/'+ 'iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_16_minimise_False.npz'
    file_path9 = data_fp + 'tnm_neuron_1/'+ 'iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_31_minimise_False.npz'
    file_path10 = data_fp + 'tnm_neuron_1/'+ 'iclr_inp_mel_best_mel_lr_0.01_rp_0.001_iter_100_seed_44_minimise_False.npz'   
    
    
    with np.load(file_path1) as fp:
            # list of np arrays
            ana_data1 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path2) as fp:
            # list of np arrays
            ana_data2 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path3) as fp:
            # list of np arrays
            ana_data3 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path4) as fp:
            # list of np arrays
            ana_data4 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path5) as fp:
            # list of np arrays
            ana_data5 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path6) as fp:
            # list of np arrays
            ana_data6 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path7) as fp:
            # list of np arrays
            ana_data7 = [fp[ele] for ele in sorted(fp.files)]
            
    with np.load(file_path8) as fp:
            # list of np arrays
            ana_data8 = [fp[ele] for ele in sorted(fp.files)]

    with np.load(file_path9) as fp:
            # list of np arrays
            ana_data9 = [fp[ele] for ele in sorted(fp.files)]
    
    with np.load(file_path10) as fp:
            # list of np arrays
            ana_data10 = [fp[ele] for ele in sorted(fp.files)]          
    
    plt.figure(figsize=(14, 8)) 
    
    # plot inputs
    plt.subplot(3, 5, 1)
    disp.specshow(normalise(ana_data1[0]), x_axis=None, hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.ylabel(r'$\bf{f_g(\tilde{z}_i)}$' '\n Freq(Hz)', multialignment='center')
    plt.title(r'$Seed = 4$')
    
    plt.subplot(3, 5, 2)
    disp.specshow(normalise(ana_data2[0]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.title(r'$Seed = 14$')
    
    plt.subplot(3, 5, 3)
    disp.specshow(normalise(ana_data3[0]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.title(r'$Seed = 16$')
    
    plt.subplot(3, 5, 4)
    disp.specshow(normalise(ana_data4[0]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.title(r'$Seed = 31$')
    
    plt.subplot(3, 5, 5)
    disp.specshow(normalise(ana_data5[0]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.title(r'$Seed = 44$')
    
    # plot max-case-onm
    plt.subplot(3, 5, 6)
    disp.specshow(normalise(ana_data1[1]), x_axis=None, hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.ylabel('$ \\bf{Maximise^1}$ \n Freq(Hz)', multialignment='center')
    
    plt.subplot(3, 5, 7)
    disp.specshow(normalise(ana_data2[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    
    plt.subplot(3, 5, 8)
    disp.specshow(normalise(ana_data3[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    
    plt.subplot(3, 5, 9)
    disp.specshow(normalise(ana_data4[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    
    plt.subplot(3, 5, 10)
    disp.specshow(normalise(ana_data5[1]), x_axis=None, hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')

    # max-case-tnm-neuron-idx-1
    plt.subplot(3, 5, 11)
    disp.specshow(normalise(ana_data6[1]), x_axis='time', hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.ylabel('$ \\bf{Maximise^2}$ \n Freq(Hz)', multialignment='center')
    plt.xlabel('Time(sec)')
    
    plt.subplot(3, 5, 12)
    disp.specshow(normalise(ana_data7[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    
    plt.subplot(3, 5, 13)
    disp.specshow(normalise(ana_data8[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    
    plt.subplot(3, 5, 14)
    disp.specshow(normalise(ana_data9[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
    
    plt.subplot(3, 5, 15)
    disp.specshow(normalise(ana_data10[1]), x_axis='time', hop_length= 315, y_axis=None, fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)')
        
    plt.subplots_adjust(wspace=0.05, hspace=0.06)
    cax = plt.axes([0.91, 0.11, 0.0125, 0.77])
    cbar = plt.colorbar(cax=cax)
    plt.savefig(data_fp+'am_comp_onm_tnm_v.pdf', bbox_inches='tight')     
    
if __name__ =='__main__':
    #fig3()
    #fig4()
    #thesis_fig_5_3()
    #thesis_fig_5_4()
    #thesis_fig_5_5()
    #thesis_fig_5_6()
    #thesis_fig_5_7()
    thesis_fig_5_8()