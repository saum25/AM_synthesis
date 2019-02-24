'''
Created on 8 Nov 2018

@author: Saumitra
'''

import argparse # parsing arguments
import tensorflow as tf
import Utils
import numpy as np
import os
import wrapper
import librosa
import pandas as pd
from collections import OrderedDict

#removes the run time tensorflow warning that points to compile code from source
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# paths of generator and classifier models and mean/std .npz
gen_model_path = 'models/prior/checkpoints/623328/623328-1170000'
cf_model_path = 'models/classifier/model1/Jamendo_augment_mel'
#cf_model_path = 'models/classifier/model2/Jamendo_augment_mel' # enable if two neuron model needs to be used.
meanstd_file_path = 'models/classifier/jamendo_meanstd.npz'

# generator parameters
gen_model_config = {
                    "batch_size" : 1, # Batch size
                    "num_frames" : 115, # DESIRED number of time frames in the spectrogram per sample at the beginning of training
                    "num_cond_frames" : 0, # Number of frames used for conditional generation
                    "num_freqs" : 80, # How many frequency bins per timestep/time frame we should model - if lower than the input feature dimension, we centre-crop the input feature!
                    'noise_dim' : 128, # Dimensionality of noise vector for the generator for each sample
                    'rnn_noise_dim_step' : 0, # ONLY FOR RNN, SET TO 0 OTHERWISE, OPTIONAL: Dimensionality noise vector additionally injected at each timestep of the RNN
                    "one_dim_conv" : False, # Whether to use 1D or 2D conv for generator/disc
                    }

def main():

    # parser for command line arguments    
    parser = argparse.ArgumentParser(description='program to synthesize examples (mel spectrograms) to maximally activate a neuron in a pre-trained CNN classifier (vocal detector)')

    # layer/neuron
    parser.add_argument('--n_out_neurons', type=int, default=1, help='number of neurons in the output layer. selects a classifier model. default: single neuron model')
    parser.add_argument('--layer', type=str, default='fc9', help='layer containing neuron of interest')
    parser.add_argument('--neuron', type=int, default=0, help='neuron idx')
    
    # optimization
    parser.add_argument('--n_iters', type=int, default=50, help='number of optimisation steps')
    parser.add_argument('--init_lr', type=float, default=1e-2, help='initial learning rate')
    parser.add_argument('--reg_param', type=float, default=1e-3, help='regularization scale factor')
    parser.add_argument('--reg_type', type=str, default='L2', help='regularizer type')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type')
    parser.add_argument('--seed', type=int, default=0, help='starting seed')
    parser.add_argument('--weight_decay', default = False, action="store_true", help='if given, forcefully makes the noise vector smaller each iteration. from Nguyen(2016)')
    parser.add_argument('--minimise', default = False, action="store_true", help='if given, the code performs activation minimisation')

    # miscellanous
    parser.add_argument('--output_dir', type=str, default=os.getcwd(), help='location to save optimisation results')
    parser.add_argument('--stats_csv', type=str, default=os.getcwd(), help='location to create a csv to store results of hyperparameter search')
    parser.add_argument('--count', type=int, default=1, help='index of hyperparamter settings')

    args = parser.parse_args()

    
    print "-------------"
    print "-------------"
    print " num out_neurons: %d" % args.n_out_neurons
    print " layer: %s" % args.layer
    print " neuron idx: %d" % args.neuron
    print "-------------"
    print " n_iters: %d" % args.n_iters
    print " init_lr: %f" % args.init_lr
    print " reg_param: %f" % args.reg_param
    print " reg_type: %s" % args.reg_type
    print " optimizer: %s" % args.optimizer
    print " seed : %d" % args.seed
    print " weight_decay: %r" %(args.weight_decay)
    print " minimisation: %r" %(args.minimise)
    print "-------------"
    print " output dir: %s" % args.output_dir
    print " hp_stats_file: %s" % args.stats_csv
    print " hp_setting_count: %d" %args.count
    print "-------------"
    print "-------------"
        
    params_dict = {
                   'out_neurons': args.n_out_neurons,
                   'layer': args.layer, 
                   'neuron': args.neuron, 
                   'reg_param': args.reg_param,
                   'reg_type':  args.reg_type, 
                   'mean_std_fp': meanstd_file_path, # good to use extra comma to prevent issues when editing later
                   }

    # miscellanous parameters
    nhop=315
    norm_flag=True
    samp_rate = 22050
    if args.optimizer !='Adam':
        final_lr = args.init_lr * 1e-8
    
    # path to store optimisation results
    results_path = args.output_dir + '/lr_' + str(args.init_lr) + '_rp_' + str(params_dict['reg_param'])
    
    # shape of generator input and output : NOTE redundancy exists due to the use of conditional RNN generator code.
    num_gen_frames = gen_model_config["num_frames"] - gen_model_config["num_cond_frames"]
    num_cond_frames = gen_model_config["num_frames"] - num_gen_frames
    noise_dim = gen_model_config["noise_dim"] + num_gen_frames * gen_model_config["rnn_noise_dim_step"]    
    shape = [gen_model_config["batch_size"], gen_model_config["num_freqs"], gen_model_config["num_frames"], 1] # Shape of input batches
    
    # setting up the noise placeholder/ or variable depending on the optimizer type
    if args.optimizer == 'Adam':
        np.random.seed(args.seed)
        with tf.variable_scope("inp"):
            inp_noise_vec = tf.Variable(np.random.normal(0.0, 1.0, size = (gen_model_config["batch_size"], noise_dim)), name = 'input_noise', dtype = tf.float32, expected_shape = (gen_model_config["batch_size"], noise_dim))
    else: # currently only vanilla-SGD
        inp_noise_vec= tf.placeholder(tf.float32, shape=[gen_model_config["batch_size"], noise_dim])
    
    print("Input noise shape: %s" %(inp_noise_vec.shape, ))

    # generate mel spectrogram excerpt
    real_batch = np.ones(shape)
    real_batch_cond = real_batch[:, :, :num_cond_frames, :]
    gen_mel = wrapper.generate_mel(inp_noise_vec, gen_model_config, real_batch_cond) # NOTE: "real_batch_cond" is a redundant variable
    print("Generator output shape:%s" %(gen_mel.shape, ))
       
    # generate activation from the desired neuron
    score = wrapper.generate_activation(gen_mel, params_dict)
    print("Score vector shape: %s" %(score.shape, ))
    
    # calculate and apply the regularisation penalty
    if (args.weight_decay == True and args.optimizer != 'Adam'): # Nguyen et al. (2016, NIPS), a high value 0.99 is used as a regulariser to reduce noise size in each iteration
        updated_score = score
        reg_penalty = tf.constant(params_dict['reg_param'], dtype=tf.float32, shape=())
    else:
        reg_penalty = wrapper.calculate_regularisation_penalty(inp_noise_vec, args.reg_type)
        updated_score = wrapper.apply_regularisation(reg_penalty, score, params_dict)
        
    print("----------------------------")

    # Calculate gradient depending on the optimizer type
    if args.optimizer == 'Adam':
        opt_obj = tf.train.AdamOptimizer(args.init_lr)
        if args.minimise==True:
            print("Activation Minimisation Case....")
            grad_vector = opt_obj.compute_gradients(updated_score, var_list = [inp_noise_vec]) # remember by default the gradient is calculated on all the variables in the collection - trainable
        else:
            print("Activation Maximisation Case....")
            grad_vector = opt_obj.compute_gradients(-1 * updated_score, var_list = [inp_noise_vec])
            
        print("Gradient vector shape: %s" %(grad_vector[0][0].shape, ))
        optm_operation = opt_obj.apply_gradients(grad_vector)
    else: # vanilla-SGD
        grad_vector = tf.gradients(updated_score, inp_noise_vec)
        print("Gradient vector shape: %s" %(grad_vector[0].shape, ))
    
    print("----------------------------")

    # Initialize all the variables # TO DO, Modify the count params function for classifier(either retrain classifier with tags or some other way), currently it's a hack.
    gen_vars = Utils.getTrainableVariables("gen")
    print("Generator Vars: " + str(Utils.getNumParams(gen_vars)))
    classifier_vars = Utils.getTrainableVariables_classifier(["gen", "inp"]) # a bit redundant but working. 
    print("Classifier Vars: " + str(Utils.getNumParams(classifier_vars)))
    restorer_gen = tf.train.Saver(gen_vars)
    restorer_cf = tf.train.Saver(classifier_vars)

    with tf.Session() as sess:

        # in 'Adam' case we need to initialise few global variables coming from Adam usage. 
        if args.optimizer == 'Adam':
            sess.run(tf.global_variables_initializer())
            
        restorer_gen.restore(sess, gen_model_path)
        print('Pre-trained generator model restored from file ' + gen_model_path)
        
        restorer_cf.restore(sess, cf_model_path)
        print('Pre-trained classifier model restored from file ' + cf_model_path)
        print("----------------------------")      
        
        # input for the vanilla SGD case
        if args.optimizer != 'Adam':
            np.random.seed(args.seed)
            z_low = np.random.normal(0.0, 1.0, size = (gen_model_config["batch_size"], noise_dim))
    
        # lists to hold elements corresponding to per optimisation iteration
        activations = []
        penalty_term = []
        grad_norm= []
        optm_stats = []

        # init the best (max or min) mel spect. we use it for inversion to audio
        best_mel = np.zeros((gen_model_config['num_freqs'], gen_model_config['num_frames']))
        
        if args.minimise==True:
            # initialise neuron activation to a very high value
            neuron_score_best = 1000.0
        else:
            # initialise neuron activation to a very low value
            neuron_score_best = -1000.0

        print
        print("Optimisation starts.....")  
        
        for iteration in range(args.n_iters):
            
            if args.optimizer == 'Adam':
                step_size = args.init_lr # just used for displaying results in each iteration
            else:
                step_size = args.init_lr + (((final_lr- args.init_lr) * iteration) / args.n_iters)       
            
            # execute the graph
            if args.optimizer == 'Adam':
                gen_output, neuron_score_iter, gradients, penalty= sess.run([gen_mel, score, grad_vector[0], reg_penalty]) # grad_vector is a list of tuples
                _= sess.run(optm_operation) #separate as this results in change of input vector
            else:
                gen_output, neuron_score_iter, gradients, penalty = sess.run([gen_mel, score, grad_vector, reg_penalty], feed_dict={inp_noise_vec : z_low}) # grad_vector is a list
                
            # save output and update score
            neuron_score_best=Utils.cond_save_mel(gen_output[0, :, :, 0], neuron_score_iter[0], neuron_score_best, iteration, results_path, args.minimise)
            best_mel = gen_output[0, :, :, 0]

            if args.weight_decay == True and args.optimizer != 'Adam':
                print("[Iteration]: %d [Neuron score (current)]: %.4f [Neuron score (O/p Saved)]: %.4f [input L2 Norm]: %.2f [Grad_mag]: %.2f [Learning Rate]: %f " %(iteration+1, neuron_score_iter[0], neuron_score_best, np.linalg.norm(z_low), np.linalg.norm(gradients[0]), step_size))
            else:    
                print("[Iteration]: %d [Neuron score (current)]: %.4f [Neuron score (O/P Saved)]: %.4f [input L2 Norm]: %.2f [Grad_mag]: %.2f [Learning Rate]: %f " %(iteration+1, neuron_score_iter[0], neuron_score_best, np.sqrt(penalty), np.linalg.norm(gradients[0]), step_size))
            
            # append neuron activations, penalty term and gradients for every iteration
            activations.append(neuron_score_iter[0])
            if args.weight_decay == True and args.optimizer != 'Adam':
                penalty_term.append(np.linalg.norm(z_low))
            else:
                penalty_term.append(np.sqrt(penalty))
            grad_norm.append(np.linalg.norm(gradients[0]))
            
            # Update the noise vector for the SGD case
            if args.optimizer != 'Adam':
                if args.minimise== True:
                    z_low = z_low - step_size * gradients[0]
                else:
                    z_low = z_low + step_size * gradients[0]
                    
                if args.weight_decay == True: # scale weights manually
                    z_low = z_low * penalty
        print
        print("Optimisation ends....")
        print('---------------------------')

        # saving plots of useful data
        print("Saving miscellanous information...")
        y_axis_param_list = [activations, penalty_term, grad_norm]
        x_axis_list = (np.arange(1, args.n_iters+1, 1)).tolist()
        path_dir = os.getcwd() + '/'+ results_path +'/'
        y_label_list = ['neuron_score', 'noise_L2_norm', 'grad_L2_norm']        
        Utils.save_misc_params(y_axis_param_list, x_axis_list, path_dir, y_label_list)
    
        # saving key stats per optimisation in a csv file as row    
        optm_stats.append(OrderedDict([('Learning Rate', args.init_lr), ('Regularisation Param', args.reg_param), ('Max/Min Activation', np.around(neuron_score_best, decimals=3)), ('Noise L2 Norm Diff', np.around(penalty_term[0] - penalty_term [-1], decimals = 3))]))
        df_stats = pd.DataFrame(optm_stats)
        if args.count==1:
            df_stats.to_csv(args.stats_csv, mode='a', index=False)
        else:
            df_stats.to_csv(args.stats_csv, mode='a', header=False, index=False)
            
        # save the best mel spectrogram to audio for auralisation                
        spect = Utils.logMelToSpectrogram(best_mel)
        audio = Utils.spectrogramToAudioFile(magnitude= spect, hopSize = nhop)
        librosa.output.write_wav(results_path+ '/'+'recon_mel_max.wav', audio, sr = samp_rate, norm=norm_flag)
        
if __name__ == "__main__":
    main()
