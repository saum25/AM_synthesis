'''
Created on 2 Jan 2019

@author: Saumitra
'''
import numpy as np
import tensorflow as tf
import models.prior.Generators
import models.classifier.Classifier

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

def post_process_gen_output(generator_output, mean, istd):
    """
    standardizes generator output by using the pre-computed mean and standard deviation vectors from the Jamendo training dataset.
    Each dimension of the mel spectrogram is standardised to mean = 0 and std dev = 1.
    @param: generator_output: generated mel spectrogram excerpt
    @param: placeholder for the mean vector
    @param: placeholder for the inverse standard deviation vector
    @return: mel_spects:standardised mel spectrogram excerpt
    """
    # TODO: Remove (if any) redundant steps
    gen_out_tranpose = tf.transpose(generator_output,(0, 2, 1, 3)) # generator output shape: 1 x 80 x 115 x 1
    mel_spects = (gen_out_tranpose [:, :, :, 0]- mean) * istd    
    mel_spects = tf.transpose(mel_spects, (0, 2, 1))
    mel_spects = tf.expand_dims(mel_spects, axis=3)    
    return mel_spects

def generate_activation(inp_excerpt, parameters, training_mode, sym_mean, sym_istd):
    """
    generates activation(s) for the desired neuron(s)/layer by passing the generated
    mel excerpt to the vocal detection classifier.
    @param: inp_excerpt: input mel spectrogram excerpt
    @param: parameters: configuration needed to generate activation
    @param: training_mode: boolean flag to indicate if the classifier runs in train mode or in test mode
    @param: sym_mean: placeholder for the mean vector
    @param: sym_istd: placeholder for the inverse standard deviation vector
    @return: activation: output of the desired neuron(s)/layer
    """
    
    # standardize mel spectrogram excerpt
    pp_mel = post_process_gen_output(inp_excerpt, sym_mean, sym_istd)
    print("Standardised generator output shape:%s" %(pp_mel.shape))
    print("----------------------------")
    
    # create classifier architecture    
    classifier_network = models.classifier.Classifier.architecture(pp_mel, training_mode)
    
    # generate score
    activation_vector = classifier_network[parameters['layer']]
    print("Activation vector shape: %s from layer <%s>" %(activation_vector.shape, parameters['layer']))
    activation = activation_vector[:, parameters['neuron']] # must be of shape (1, ) as it's the score of a single neuron
    return activation

def calculate_regularisation_penalty(latent_vector, reg_type):
    """
    Calculates the penalisation penalty for the latent vector based on the regularisation type.
    @param: latent_vector: input noise vector
    @param: reg_type: regularisation type
    @return: : returns a scalar indicating the amount of regularisation
    """

    lat_vec_len = latent_vector.get_shape().as_list()[1]
    
    if reg_type != "No_Regularisation":
        # two choices either L2-norm or Gaussian prior
        if reg_type == "L2":
            return (tf.norm(latent_vector, axis=1))
        else: # TO DO
            mean = tf.constant(0., tf.float32, (lat_vec_len, ))
            cov = tf.constant(np.identity(lat_vec_len, dtype=np.float32), tf.float32)
            print("[Gaussian Prior] Mean vec shape: %s Cov matrix shape: %s" %(mean.shape, cov.shape))
            rv = tf.contrib.distributions.MultivariateNormalFullCovariance(mean, cov)
            return(rv.prob(latent_vector))
    else:
        return tf.constant(0, tf.float32, ())
    
def apply_regularisation(params_dict, reg_penalty, score):
    """
    updates the activation value by applying the regularisation penalty
    @param: params_dict: parameter dictionary
    @param: reg_penalty: regularisation penalty
    @param: score: activation value from the desired neuron
    @return: updated_score: updated activation value
    """
    if params_dict["reg_type"]=="Gaussian_Prior": # TO DO
        updated_score = tf.add(score, tf.multiply(tf.constant(params_dict['reg_param'], tf.float32), reg_penalty))
    else:
        updated_score = tf.subtract(score, tf.multiply(tf.constant(params_dict['reg_param'], tf.float32), reg_penalty)) # takes care L2 and no regulariser case (reg_penatly = 0)
    
    return updated_score

