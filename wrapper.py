'''
Created on 2 Jan 2019

@author: Saumitra
'''
import tensorflow as tf
import models.prior.Generators
import models.classifier.Classifier
import Utils

def generate_mel(inp_noise_vec, model_config, real_batch_cond):
    '''
    maps input noise vector to mel spectrogram
    @param: inp_noise_vec: input noise vector (shape: ? x 128)
    @param: model_config: generator configuration dictionary
    @param: real_batch_cond: redundant variable, using to prevent the need to change the generator class definition
    @return: gen_out: generated mel spectrogram excerpt (shape: ? x 80 x 115 x 1) consisting of 80 mel bands and 115 time frames
    '''
    gen_func = models.prior.Generators.ConvGenerator(model_config)
    gen_out = gen_func.get_output(inp_noise_vec, real_batch_cond, reuse=False)
    return gen_out

def post_process_gen_output(generator_output, mean_std_fp):
    """
    standardizes generator output by using the pre-computed mean and standard deviation vectors from the Jamendo training dataset.
    Each dimension of the mel spectrogram is standardised to mean = 0 and std dev = 1.
    @param: generator_output: generated mel spectrogram excerpt
    @param: mean_std_fp: file path for the mean and std dev .npz file
    @return: mel_spects: standardised mel spectrogram excerpt
    """
    # mean and std dev arrays
    mean, istd = Utils.read_meanstd_file(mean_std_fp)
    sym_mean = tf.constant(mean, dtype= tf.float32)
    sym_istd = tf.constant(istd, dtype=tf.float32) 

    gen_out_tranpose = tf.transpose(generator_output,(0, 2, 1, 3)) # generator output shape: 1 x 80 x 115 x 1
    mel_spects = (gen_out_tranpose [:, :, :, 0]- sym_mean) * sym_istd    
    mel_spects = tf.transpose(mel_spects, (0, 2, 1))
    mel_spects = tf.expand_dims(mel_spects, axis=3)    
    return mel_spects

def generate_activation(inp_excerpt, parameters):
    """
    generates activation(s) for the desired neuron(s)/layer by passing the generated
    mel excerpt to the vocal detection classifier.
    @param: inp_excerpt: input mel spectrogram excerpt
    @param: parameters: configuration needed to generate activation
    @return: activation: output of the desired neuron(s)/layer
    """
    
    # standardize mel spectrogram excerpt
    pp_mel = post_process_gen_output(inp_excerpt, parameters['mean_std_fp'])
    print("Standardised generator output shape:%s" %(pp_mel.shape))

    # training model flag
    training_mode = tf.constant(False)
    
    # create classifier architecture    
    classifier_network = models.classifier.Classifier.architecture(pp_mel, training_mode, parameters['out_neurons'])
    
    # generate score
    activation_vector = classifier_network[parameters['layer']]
    print("Activation vector shape: %s from layer <%s>" %(activation_vector.shape, parameters['layer'])) # first dim: n_batches, second dim: n_neurons
    activation = activation_vector[:, parameters['neuron']] # must be of shape (1, ) as it's the score of a single neuron
    return activation

def calculate_regularisation_penalty(latent_vector, reg_type):
    """
    Calculates the penalisation penalty for the latent vector based on the regularisation type.
    @param: latent_vector: input noise vector
    @param: reg_type: regularisation type
    @return: : returns a scalar indicating the amount of regularisation
    """    
    # two choices: L2-norm or Gaussian prior - conceptually both aim to do same.
    if reg_type == "L2":
        return (tf.reduce_sum(tf.square(latent_vector)))
    else: # TO DO
        lat_vec_len = latent_vector.get_shape().as_list()[1]
        mean = tf.constant(0., tf.float32, (lat_vec_len, ))
        print("[Gaussian Prior] Mean vec shape: %s" %(mean.shape))
        rv = tf.contrib.distributions.MultivariateNormalDiag(mean)
        return(rv.prob(latent_vector))

    
def apply_regularisation(reg_penalty, score, params_dict):
    """
    updates the activation value by applying the regularisation penalty
    @param: reg_penalty: regularisation penalty
    @param: score: activation value from the desired neuron
    @param: params_dict: parameter dictionary    
    @return: updated_score: updated activation value
    """
    if params_dict["reg_type"]=="Gaussian_Prior": # TO DO
        updated_score = tf.add(score, tf.multiply(tf.constant(params_dict['reg_param'], tf.float32), reg_penalty))
    else:
        updated_score = tf.subtract(score, tf.multiply(tf.constant(0.5 * params_dict['reg_param'], tf.float32), reg_penalty))
    
    return updated_score

