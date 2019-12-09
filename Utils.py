'''
Created on 8 Nov 2018

@author: Saumitra
'''

import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import librosa.display as disp
import os
import numpy.linalg as linalg
import librosa

def non_local_block(inputs):
    input_shape = inputs.get_shape().as_list()
    assert(len(input_shape) == 3)
    features = input_shape[2]
    assert(features % 2 == 0)
    theta = tf.layers.conv1d(inputs, features//2, 1, padding="same", use_bias=False)
    phi = tf.layers.conv1d(inputs, features//2, 1, padding="same", use_bias=False)
    g = tf.layers.conv1d(inputs, features//2, 1, padding="same", use_bias=False)

    # Compute similarity matrix
    f = tf.matmul(phi, tf.transpose(theta, [0, 2, 1]))
    f = tf.nn.softmax(f, axis=2)

    # compute output path
    y = tf.matmul(f, g)

    # reshape to input tensor format
    #y = tf.reshape(y, [input_shape[0], input_shape[1], features // 2])

    # project filters
    y = tf.layers.conv1d(y, features, 1, padding="same", use_bias=False)

    # residual connection
    residual = tf.add(inputs, y)

    return residual

def getTrainableVariables(tag=""):
    return [v for v in tf.trainable_variables() if tag in v.name]

def getTrainableVariables_classifier(tag_list):
    classifier_vars = []
    for v in tf.trainable_variables():
        if (tag_list[0] not in v.name) and (tag_list[1] not in v.name):
            classifier_vars.append(v)
        else:
            pass
    return classifier_vars

def getNumParams(tensors):
    return np.sum([np.prod(t.get_shape().as_list()) for t in tensors])

def pad_freqs(tensor, target_shape):
    '''
    Pads the frequency axis of a 4D tensor of shape [batch_size, freqs, timeframes, channels] or 2D tensor [freqs, timeframes] with zeros
    so that it reaches the target shape. If the number of frequencies to pad is uneven, the rows are appended at the end. 
    :param tensor: Input tensor to pad with zeros along the frequency axis
    :param target_shape: Shape of tensor after zero-padding
    :return: 
    '''
    target_freqs = (target_shape[1] if len(target_shape) == 4 else target_shape[0])
    if isinstance(tensor, tf.Tensor):
        input_shape = tensor.get_shape().as_list()
    else:
        input_shape = tensor.shape

    if len(input_shape) == 2:
        input_freqs = input_shape[0]
    else:
        input_freqs = input_shape[1]

    diff = target_freqs - input_freqs
    if diff % 2 == 0:
        pad = [(diff/2, diff/2)]
    else:
        pad = [(diff//2, diff//2 + 1)] # Add extra frequency bin at the end

    if len(target_shape) == 2:
        pad = pad + [(0,0)]
    else:
        pad = [(0,0)] + pad + [(0,0), (0,0)]

    if isinstance(tensor, tf.Tensor):
        return tf.pad(tensor, pad, mode='constant', constant_values=0.0)
    else:
        return np.pad(tensor, pad, mode='constant', constant_values=0.0)

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)


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

def save_mel(gen_out, directory, score, iteration=0, pred=0, case='synth'):
    '''
    normalise (0-1) and save output representation from the generator model
    @param: gen_out: unnormalised generator output
    @param: directory: path to save results
    @param: score: activation for the current iteration
    @param: iteration: optimisation iteration count
    @param: pred: prediction the model applies to the instance
    @param: case: instance to save is synthesised from AM or selected from datset
    @return: NA
    '''
    plt.figure(figsize=(6, 4))
    #disp.specshow(normalise(gen_out), x_axis = 'time', y_axis='mel', sr=22050, hop_length=315, fmin=27.5, fmax=8000, cmap = 'coolwarm')
    disp.specshow((gen_out), x_axis = 'time', y_axis='mel', sr=22050, hop_length=315, fmin=27.5, fmax=8000, cmap = 'coolwarm') # remove the normalisation to understand value distribution
    plt.title('mel spectrogram of GAN output')
    plt.tight_layout()
    plt.colorbar()
    if case == 'dataset':
        plt.savefig(directory+'/'+'mel_score_'+"%.6f" %score+ '_pred_'+"%.6f" %pred + '.pdf', dpi=300)
    elif case == 'synth':
        plt.savefig(os.getcwd() + '/'+ directory +'/'+'examples/'+ 'example_iteration'+ str(iteration) + '_score' + str(round(score, 2)) +'.pdf', dpi = 300)
    else:
        raise ValueError('%s is not a valid option for case' %case)
    plt.close()

def cond_save_mel(gen_out, best_mel_spect, score, best_score, n_iter, dir_path, min_flag):
    """
    # updates best_score and saves the corresponding GAN output only if there is a substantial change in the activation score
    @param: gen_out: GAN output
    @param: best_mel_spect: best mel spectrogram
    @param: score: neuron activation for the interation n_iter
    @param: best_score: best value of score so far
    @param: n_iter: optimisation iteration number
    @param: dir_path: location to save the GAN output
    @param: min_flag: if True, it's activation minimisation else maximisation
    @return: updated value of best_score
    """
    if (((min_flag and (score < best_score)) or ((not min_flag) and (score > best_score))) and (np.trunc(np.abs((np.abs(score) - np.abs(best_score)) * 10)) >=1)):
        print("Saving example_iteration_%d...." %(n_iter+1))
        #save_mel(gen_out, dir_path, score, iteration=n_iter+1)
        return gen_out, score
    else:
        return best_mel_spect, best_score

def save_misc_params(y_axis_param, x_axis_param, output_dir, y_axis_label):
    """
    Saves a plot depicting how the parameter represented by y_axis_param changes w.r.t. param represented by x_axis_param
    @param: y_axis_param: list of params lists
    @param: x_axis: list indicating iteration count
    @param: output_dir: path to the output directory
    @param: y_axis_label: label for each item in the param list
    @return: NA
    """
    for param_idx, param in enumerate(y_axis_param):
        plt.figure(figsize=(6,4))
        plt.plot(x_axis_param, param)
        plt.xticks(np.linspace(1, len(param), 10, dtype=np.int32)) # x-axis will have 10 ticks, linearly spaced
        plt.xlabel("iterations")
        plt.ylabel(y_axis_label[param_idx])
        plt.savefig(output_dir + y_axis_label[param_idx] +'.pdf', dpi = 300)
        plt.close()

def crop(tensor, target_shape, match_feature_dim=True):
    '''
    Crops a 3D tensor [batch_size, width, channels] along the width axes to a target shape.
    Performs a centre crop. If the dimension difference is uneven, crop last dimensions first.
    :param tensor: 4D tensor [batch_size, width, height, channels] that should be cropped.
    :param target_shape: Target shape (4D tensor) that the tensor should be cropped to
    :return: Cropped tensor
    '''
    shape = np.array(tensor.get_shape().as_list())
    diff = shape - np.array(target_shape)
    assert(diff[0] == 0)# Only width axis can differ
    if (diff[1] % 2 != 0):
        print("WARNING: Cropping with uneven number of extra entries on one side")
    assert diff[1] >= 0 # Only positive difference allowed
    if diff[2] == 0:
        return tensor
    crop_start = diff // 2
    crop_end = diff - crop_start

    return tensor[:,:,crop_start[2]:-crop_end[2],:]

def read_meanstd_file(file_path):
    """
    load mean and std dev per dimension (frequency band) calculated over the Jamendo training data
    @param: file path to the mean std dev file
    @return: mean: mean across each freq band
    @return: istd: inverse of std dev across each freq band
    """   
    with np.load(file_path) as f:
        mean = f['mean']
        std = f['std']      
    istd = np.reciprocal(std)
    
    return mean, istd

def save_max_activation(lr_list, max_act_list, op_dir):
    """
    saves the plot of maximum activation per setting of hyperparameters
    @param: lr_list: list of initial learning rates
    @param: max_act_list: maximum activation values per setting of hyperparameters
    @param: op_dir: output directory
    @return: NA
    """
    
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1, len(lr_list)+1, 1), max_act_list)
    plt.xticks(np.arange(1, len(lr_list)+1, 1), ['{:f}'.format(temp) for temp in lr_list])
    plt.xlabel("learning rate")
    plt.ylabel('maximum activation')
    plt.savefig(os.getcwd() + '/'+ op_dir+ '/' + 'max_act' +'.pdf', dpi = 300)
    
def create_mel_filterbank(sample_rate, frame_len, num_bands, min_freq,
                          max_freq):
    """
    @ From Jan Schluter's implementation for ISMIR 2015 paper
    Creates a mel filterbank of `num_bands` triangular filters, with the first
    filter starting at `min_freq` and the last one stopping at `max_freq`.
    Returns the filterbank as a matrix suitable for a dot product against
    magnitude spectra created from samples at a sample rate of `sample_rate`
    with a window length of `frame_len` samples.
    """
    # prepare output matrix
    input_bins = (frame_len // 2) + 1
    filterbank = np.zeros((input_bins, num_bands))

    # mel-spaced peak frequencies
    min_mel = 1127 * np.log1p(min_freq / 700.0)
    max_mel = 1127 * np.log1p(max_freq / 700.0)
    spacing = (max_mel - min_mel) / (num_bands + 1)
    peaks_mel = min_mel + np.arange(num_bands + 2) * spacing
    peaks_hz = 700 * (np.exp(peaks_mel / 1127) - 1)
    fft_freqs = np.linspace(0, sample_rate / 2., input_bins)
    peaks_bin = np.searchsorted(fft_freqs, peaks_hz)

    # fill output matrix with triangular filters
    for b, filt in enumerate(filterbank.T):
        # The triangle starts at the previous filter's peak (peaks_freq[b]),
        # has its maximum at peaks_freq[b+1] and ends at peaks_freq[b+2].
        left_hz, top_hz, right_hz = peaks_hz[b:b+3]  # b, b+1, b+2
        left_bin, top_bin, right_bin = peaks_bin[b:b+3]
        # Create triangular filter compatible to yaafe
        filt[left_bin:top_bin] = ((fft_freqs[left_bin:top_bin] - left_hz) /
                                  (top_bin - left_bin))
        filt[top_bin:right_bin] = ((right_hz - fft_freqs[top_bin:right_bin]) /
                                   (right_bin - top_bin))
        filt[left_bin:right_bin] /= filt[left_bin:right_bin].sum()

    return filterbank

    
def spectrogramToLogMel(spectrogram, sample_rate=22050, fftWindowSize=1024, filterbank=None):
    # Parameters for Mel spectrogram calculation
    mel_bands = 80
    mel_min = 27.5
    mel_max = 8000
    bin_nyquist = fftWindowSize // 2 + 1
    bin_mel_max = bin_nyquist * 2 * mel_max // sample_rate

    # prepare mel filterbank
    if filterbank is None:
        filterbank = create_mel_filterbank(sample_rate, fftWindowSize, mel_bands, mel_min, mel_max)
        filterbank = filterbank[:bin_mel_max]
        return filterbank

    # compute log mel spectrogram matrix
    spect_T = spectrogram.T
    mspect = np.dot(spect_T[:, :bin_mel_max], filterbank).astype(np.float32)
    mspect = np.log(np.maximum(mspect, 1e-7)) # Normalisation like in Schlueters classifier. Puts it into range [log(1e-7), +x]
    #mspect = np.log1p(mspect) # [0, +x]
    return mspect.T
    
def logMelToSpectrogram(melspec, sample_rate=22050, fftWindowSize=1024):
    # Invert the log normalization
    melspec = np.exp(melspec)

    # Compute Mel filterbank
    filterbank = spectrogramToLogMel(None, sample_rate, fftWindowSize)

    # Invert Mel filterbank
    filterbank_pinv = linalg.pinv(filterbank) # Should be 80x372

    # Invert Mel spectrogram with inverted filterbank
    spect = np.dot(melspec.T, filterbank_pinv).T  # (freqs, timeframes)

    # Set negative spectrogram magnitudes to zero
    spect = np.maximum(spect, 0.0)

    # Append zeros for higher frequencies that we left out when computing the Mel spectrogram
    freqs = fftWindowSize // 2 + 1
    pad_freqs = freqs - spect.shape[0]
    spect = np.pad(spect, [(0,pad_freqs), (0, 0)], mode='constant', constant_values=0)

    return spect

def reconPhase(magnitude, fftWindowSize, hopSize, phaseIterations=10, initPhase=None, length=None):
    '''
    Griffin-Lim algorithm for reconstructing the phase for a given magnitude spectrogram, optionally with a given
    intial phase.
    :param magnitude:
    :param fftWindowSize:
    :param hopSize:
    :param phaseIterations:
    :param initPhase:
    :param length:
    :return:
    '''
    for i in range(phaseIterations):
        if i == 0:
            if initPhase is None:
                reconstruction = np.random.random_sample(magnitude.shape) + 1j * (2 * np.pi * np.random.random_sample(magnitude.shape) - np.pi)
            else:
                reconstruction = np.exp(initPhase * 1j) # e^(j*phase), so that angle => phase
        else:
            reconstruction = librosa.core.stft(audio, fftWindowSize, hopSize)
        spectrum = magnitude * np.exp(1j * np.angle(reconstruction))
        if i == phaseIterations - 1:
            audio = librosa.core.istft(spectrum, hopSize, length=length)
        else:
            audio = librosa.core.istft(spectrum, hopSize)
    return audio

def spectrogramToAudioFile(magnitude, hopSize, fftWindowSize=1024, phaseIterations=100, phase=None, length=None):
    '''
    Computes an audio signal from the given magnitude spectrogram, and optionally an initial phase.
    Griffin-Lim is executed to recover/refine the given the phase from the magnitude spectrogram.
    :param magnitude:
    :param fftWindowSize:
    :param hopSize:
    :param phaseIterations:
    :param phase:
    :param length:
    :return:
    '''
    if phase is not None:
        if phaseIterations > 0:
            # Refine audio given initial phase with a number of iterations
            return reconPhase(magnitude, fftWindowSize, hopSize, phaseIterations, phase, length)
        # reconstructing the new complex matrix
        stftMatrix = magnitude * np.exp(phase * 1j) # magnitude * e^(j*phase)
        audio = librosa.core.istft(stftMatrix, hop_length=hopSize, length=length)
    else:
        audio = reconPhase(magnitude, fftWindowSize, hopSize, phaseIterations)
    return audio

def save_audio(melspect, path, activation, prediction, hopsize, iterations=100, norm_flag=True):
    """
    invert input mel spectrogram, generate phase (using Griff & Lim)
    and save audio.
    @param: melspect: input mel spectrogram
    @param: path: directory path to save audio
    @param: activation: neuron score for the input mel spectrogram
    @param: prediction: model prediction for the input mel spectrogram
    @param: hopsize: hop length (samples)
    @param: iterations: number of iterations for phase reconstruction
    @param: norm_flag: indicates to normalise (-1 to +1) audio before saving. True-> normalise, False-> don't normalise
    """
    
    spect = logMelToSpectrogram(melspect) # expects data in shape 80 x 115
    audio = spectrogramToAudioFile(spect, hopsize, phaseIterations=iterations)
    name = path+'/'+'recon_mel_score_'+'%.6f' %activation+'_pred_'+'%.6f'%prediction+'.wav' # string formatting is used here to truncate-> Need to find a better way?
    librosa.output.write_wav(name, audio, sr=22050, norm=norm_flag)
    
    
    
    