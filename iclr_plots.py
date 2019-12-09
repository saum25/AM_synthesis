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
    
    
if __name__ =='__main__':
    #fig3()
    fig4()
