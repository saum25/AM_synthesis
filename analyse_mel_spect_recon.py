'''
Created on 3 Feb 2019

Code to invert spectrogram and mel spectrogram
to audio for verification of inversion and phase
generation code we use in the AM pipeline.There are two cases, 
in the first we read an audio (speech, music)
using librosa, generate spectrograms and mel spectrograms, invert
them in time domain. We also analysed if the same performance is
obtained for inverting mels from Jan's code. So, we save N mel
using predict.py from SVD-TF code and invert it.
@author: Saumitra
'''

import librosa
import numpy as np
import librosa.display as disp
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import Utils

#audio_path='../deep_inversion/datasets/jamendo/audio/05 - Elles disent.mp3'
audio_path='../deep_inversion/datasets/jamendo/audio/03 - Say me Good Bye.mp3'
#audio_path='../deep_inversion/datasets/jamendo/audio/mix.wav'
#audio_path='results/audio_recon_exp/inputs/LA_T_5459362.wav' # speech file
dump_path = './results/audio_recon_exp/' # save results to

nfft= 1024
nhop= 315
bin_mel_max= 372
samp_rate= 22050
nfilts = 80
fmin=27.5
fmax=8000
off= 55#0.55 # for speech case set = 0.0
dur= 10#1.65 # for speech case set = None
fs= 7
full_range = False # defines if mel filters need to be designed between some range
norm_flag= True
invert_Jan_Mel= True


if invert_Jan_Mel == True:
    # load pre-processed(log magnitude, mean=0, std=1), pre-saved mel spectrogram
    # mel spect to invert
    index=900

    # load training data set-wise mean and std dev
    with np.load('models/classifier/jamendo_meanstd.npz') as f:
        mean = f['mean']
        std = f['std']
 
    # load mel spects    
    with np.load('results/audio_recon_exp/inputs/Jan_mel_SayMeGoodBye_1000MelExcerpts.npz') as fd:
        mel_spects = fd['arr_0']
        mel_spect = mel_spects[index]

    print("[Shape] extracted mel spect:"),
    print mel_spect.shape
    
    # unnormalise the mel spect
    mel_spect = (mel_spect*std)+mean
    
    # create mel filts
    mel_filts=Utils.create_mel_filterbank(sample_rate=samp_rate, frame_len=nfft, num_bands=nfilts, min_freq=fmin, max_freq=fmax)
    print("[Shape] mel filters:"),
    print mel_filts.shape
    
    # map mel spectrogram to spectrogram
    # to do this we use pseudo inverse of mel filter bank matrix
    mel_to_spect=np.dot(np.exp(mel_spect), np.linalg.pinv(mel_filts[:bin_mel_max]))
    print "[Shape] inverted mel to spect:",
    print mel_to_spect.shape
    
    # set negative values to 0. Such values arise due to the presence of them in the pseudo inverse matrix
    np.maximum(mel_to_spect.T, 0.0)
    # fill the gap between n_fft//2 + 1 and bin_mel_max by 0
    pad_freqs = (nfft//2+1) - (mel_to_spect.T).shape[0]
    updated_mel_to_spect = np.pad(mel_to_spect.T, [(0,pad_freqs), (0, 0)], mode='constant', constant_values=0)
    print "[Shape] updated inverted mel to spect:",
    print updated_mel_to_spect.shape
else:
    #read audio
    audio_buffer, SR = librosa.core.load(audio_path, sr = samp_rate, offset=off, duration=dur)
    librosa.output.write_wav(dump_path+'inp.wav', audio_buffer, sr=SR, norm=norm_flag)

    # stft
    mag = np.abs(librosa.core.stft(audio_buffer, n_fft=nfft, hop_length=nhop))
    print("[Shape] spect:"),
    print mag.shape
    plt.figure()
    plt.subplot(3, 1, 1)
    disp.specshow(librosa.amplitude_to_db(mag, ref=np.max, top_db=80), sr=SR, hop_length=nhop, x_axis='off', y_axis='linear', cmap='coolwarm')
    plt.ylabel('Hz', fontsize = fs, labelpad = 1)
    plt.yticks(fontsize = fs)
    plt.title('Input', fontsize=fs)

    #invert the stft to audio
    ab_spect_recon = Utils.spectrogramToAudioFile(magnitude=mag, hopSize= nhop)
    librosa.output.write_wav(dump_path+'inp_spect_recon.wav', ab_spect_recon, sr=SR, norm=norm_flag)
    
    # stft of the inverted audio
    mag_spect_recon_spect = np.abs(librosa.core.stft(ab_spect_recon, n_fft=nfft, hop_length=nhop))
    plt.subplot(3, 1, 2)
    disp.specshow(librosa.amplitude_to_db(mag_spect_recon_spect, ref=np.max), sr=SR, hop_length=nhop, x_axis='off', y_axis='linear', cmap='coolwarm')
    plt.ylabel('Hz', fontsize = fs, labelpad = 1)
    plt.yticks(fontsize = fs)
    plt.title('Inverted spectrogram', fontsize=fs)

    # design mel filters
    if full_range==True:
        mel_filts = librosa.filters.mel(sr=SR, n_fft=nfft, n_mels=80)
    else:
        mel_filts = librosa.filters.mel(sr=SR, n_fft=nfft, n_mels=80, fmin=27.5, fmax=8000)

    print("[Shape] mel filters:"),
    print mel_filts.shape

    # spect to mel spectafter matrix mult and post process
    mel_spect = np.log(np.maximum(np.dot(mel_filts, mag), 1e-7))
    print "[Shape] mel spectrogram:",
    print mel_spect.shape    

    # invert mel->spect
    mel_to_spect=np.dot(np.linalg.pinv(mel_filts), np.exp(mel_spect))
    print "[Shape] inverted mel to spect:",
    print mel_to_spect.shape

    # pinv matrix results in some negative values
    np.maximum(mel_to_spect, 0.0)
    # for consistency
    updated_mel_to_spect = mel_to_spect

# invert spectrogram to audio
ab_mel_recon = Utils.spectrogramToAudioFile(magnitude=updated_mel_to_spect, hopSize= nhop)
librosa.output.write_wav(dump_path+'inp_mel_recon.wav', ab_mel_recon, sr=samp_rate, norm=norm_flag)

# power spectrogram of the mel inverted audio
mag_spect_recon_mel = np.abs(librosa.core.stft(ab_mel_recon, n_fft=nfft, hop_length=nhop))
plt.subplot(3, 1, 3)
disp.specshow(librosa.amplitude_to_db(mag_spect_recon_mel, ref=np.max, top_db=80), sr=samp_rate, hop_length=nhop, x_axis='time', y_axis='linear', cmap='coolwarm')
plt.yticks(fontsize = fs)
plt.xticks(fontsize = fs)
plt.ylabel('Hz', fontsize = fs, labelpad = 1)
plt.xlabel('Time', fontsize = fs, labelpad = 1)
plt.title('Inverted mel spectrogram', fontsize=fs)

# manually set gap between subplots
# https://jdhao.github.io/2017/06/11/mpl_multiplot_one_colorbar/
plt.subplots_adjust(bottom=0.1, left=0.09, right=0.8, top=0.9)

# define the location of cbar, e.g., I think 0.82-> put cbar at location 0.82 in the figure
cax = plt.axes([0.82, 0.1, 0.025, 0.8])
cbar = plt.colorbar(cax=cax, format='%+2.0f dB')
# change cbar label size
cbar.ax.tick_params(labelsize=fs)

# adding super title but may be removed in the final version
plt.suptitle('Power spectrograms')
plt.savefig(dump_path+'plots.pdf', dpi=300)






