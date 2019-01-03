'''
Created on 8 Nov 2018

@author: Saumitra
'''

import argparse # parsing arguments
import models.prior.Generators
import models.classifier.Classifier
import tensorflow as tf
import Utils
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import librosa.display as disp
import os
import wrapper

#removes the run time tensorflow warning that points to compile code from source
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Paths of generator and classifier model and mean std files
gen_model_path = 'models/prior/checkpoints/623328/623328-1170000'
cf_model_path = 'models/classifier/model3/Jamendo_augment_mel'
meanstd_file_path = 'models/classifier/jamendo_meanstd.npz'

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
    parser = argparse.ArgumentParser(description='Program to synthesize examples that maximally activate neuron(s)/layer in a pre-trained CNN classifier')

    # layer/neuron
    parser.add_argument('--layer', type=str, default='act_out_fc8', help='CNN layer containing neuron(s) of interest')
    parser.add_argument('--neuron', type=int, default=0, help='neuron(s) to maximally activate')
    
    # optimization
    parser.add_argument('--n_iters', type=int, default=50, help='number of iterations')
    parser.add_argument('--init_lr', type=float, default=1e-2, help='initial learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-10, help='final learning rate')
    parser.add_argument('--reg_param', type=float, default=1e-2, help='regularization scale factor')
    parser.add_argument('--reg_type', type=str, default='L2', help='regularizer type')
    
    # miscellanous
    parser.add_argument('--output_dir', type=str, default=os.getcwd(), help='location to save results')
    
    args = parser.parse_args()
    
    print "-------------"
    print " layer: %s" % args.layer
    print " neuron idx: %s" % args.neuron
    print "-------------"
    print " n_iters: %s" % args.n_iters
    print " init_lr: %s" % args.init_lr
    print " final_lr: %s" % args.final_lr
    print " reg_param: %s" % args.reg_param
    print " reg_type: %s" % args.reg_type
    print "-------------"
    print " output dir: %s" % args.output_dir
    print "-------------"
    
    params_dict = {
                   'layer': args.layer,
                   'neuron': args.neuron,
                   'iterations': args.n_iters,
                   'start_step_size': args.init_lr,
                   'end_step_size': args.final_lr,
                   'reg_param': args.reg_param,
                   'reg_type':  args.reg_type, # good to use extra comma to prevent issues when editing later
                   'mean_std_fp': meanstd_file_path,
                   }
    # Determine the shape of generator input and output : CAUTION redundancy exists due to the use of conditional RNN generator code.
    num_gen_frames = gen_model_config["num_frames"] - gen_model_config["num_cond_frames"]
    num_cond_frames = gen_model_config["num_frames"] - num_gen_frames
    noise_dim = gen_model_config["noise_dim"] + num_gen_frames * gen_model_config["rnn_noise_dim_step"]    
    shape = [gen_model_config["batch_size"], gen_model_config["num_freqs"], gen_model_config["num_frames"], 1] # Shape of input batches
    
    # setting up the noise placeholder
    inp_noise_vec= tf.placeholder(tf.float32, shape=[gen_model_config["batch_size"], noise_dim])
    print("Input noise shape: %s" %(inp_noise_vec.shape, ))
    real_batch = np.ones(shape)
    real_batch_cond = real_batch[:, :, :num_cond_frames, :]
    
    # generate mel spectrogram excerpt
    gen_mel = wrapper.generate_mel(inp_noise_vec, gen_model_config, real_batch_cond) # CAUTION: "real_batch_cond" is a redundant variable
    print("Generator output shape:%s" %(gen_mel.shape, ))
       
    # generate desired neuron(s)/layer activation
    mean, istd = Utils.read_meanstd_file(params_dict['mean_std_fp'])
    sym_mean = tf.constant(mean, dtype= tf.float32)
    sym_istd = tf.constant(istd, dtype=tf.float32) 
    training_mode = tf.constant(False)
    score = wrapper.generate_activation(gen_mel, params_dict, training_mode, sym_mean, sym_istd)
    print("Score vector shape: %s" %(score.shape, ))
    
    # calculate the regularisation penalty
    reg_penalty = wrapper.calculate_regularisation_penalty(inp_noise_vec, args.reg_type)
    
    # Update the score depending on the regularisation type
    updated_score = wrapper.apply_regularisation(params_dict, reg_penalty, score)
    
    # Calculate gradient
    grad_vector = tf.gradients(updated_score, inp_noise_vec)
    print("Gradient vector shape: %s" %(grad_vector[0].shape, ))
    print("----------------------------")

    # Execute the dataflow graph # TO DO, Modify the count params function for classifier(either retrain classifier with tags or some other way), currently it's a hack.
    gen_vars = Utils.getTrainableVariables("gen")
    print("Generator Vars: " + str(Utils.getNumParams(gen_vars)))
    classifier_vars = Utils.getTrainableVariables_classifier("gen")
    print("Classifier Vars: " + str(Utils.getNumParams(classifier_vars)))
    restorer_gen = tf.train.Saver(gen_vars)
    restorer_cf = tf.train.Saver(classifier_vars)

    sess = tf.Session()    
    #sess.run(tf.global_variables_initializer()) # Don't think it's needed. So removing it. Experimentally also I don't see any effect.
        
    restorer_gen.restore(sess, gen_model_path)
    print('Pre-trained generator model restored from file ' + gen_model_path)
    
    restorer_cf.restore(sess, cf_model_path)
    print('Pre-trained classifier model restored from file ' + cf_model_path)
    print("----------------------------")
    
    # list of different learning rates
    start_lr_list = [params_dict['start_step_size']]#[10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6] # bypasses the init_lr argument during grid search
    
    for lr in start_lr_list:

        print
        print("Optimisation starts.....")        
        # input
        np.random.seed(0)
        z_low = np.random.normal(0.0, 1.0, [gen_model_config["batch_size"], inp_noise_vec.get_shape().as_list()[1]])

        # each element corresponds to per iteration for a selected set of hyperparameters
        activations = []
        penalty_term = []
        grad_norm_list = []
        
        # initialise neuron activation to a very low value
        neuron_score_max = np.array([-100])
        
        # bypass the start and end step size parameters for grid search
        params_dict['start_step_size'] = lr
        params_dict['end_step_size'] = lr * 1e-8
        
        print("--------------Learning rate: %f--------------" %(params_dict['start_step_size']))
        
        for iteration in range(params_dict['iterations']):
            max_flag = 0 # reinit in every iteration
            
            step_size = params_dict['start_step_size'] + ((params_dict['end_step_size'] - params_dict['start_step_size']) * iteration) / params_dict['iterations']        
            
            # execute the graph
            gen_output, neuron_score_iter, gradients, penalty = sess.run([gen_mel, score, grad_vector, reg_penalty], feed_dict={inp_noise_vec : z_low})
                
            if (neuron_score_iter > neuron_score_max) and (np.trunc(np.abs((np.abs(neuron_score_iter) - np.abs(neuron_score_max)) * 100)) >=1):
                neuron_score_max = neuron_score_iter
                max_flag = 1
                #print("Max Neuron Score: %f" %(neuron_score_max))
            
            print("[Iteration]: %d [Neuron score (current)]: %.4f [Neuron score (Max)]: %.4f [Penalty]: %.2f [Grad_mag]: %.2f [Learning Rate]: %f " %(iteration+1, neuron_score_iter, neuron_score_max, penalty, np.linalg.norm(gradients[0]), step_size))
            
            # save generator output
            if max_flag:
                print("Saving example_iteration%d...." %(iteration+1))
                Utils.save_gen_out(gen_output[0, :, :, 0], iteration+1, args.output_dir + '/lr_' + str(lr), neuron_score_max[0])
    
            # Update the noise vector        
            z_low = z_low + step_size * gradients[0] # simple gradient ascent: TO DO: Try other methods of optimisation
            
            # append neuron activations, penalty term and gradients for every iteration
            activations.append(neuron_score_iter)
            penalty_term.append(penalty)
            grad_norm_list.append(np.linalg.norm(gradients[0]))
                    
        # saving plots of other useful data
        y_axis_param_list = [activations, penalty_term, grad_norm_list]
        x_axis_list = (np.arange(1, params_dict['iterations']+1, 1)).tolist()
        path_dir = os.getcwd() + '/'+ args.output_dir + '/lr_' + str(lr) +'/'
        y_label_list = ['neuron_score', 'penalty_term', 'grad_norm']
        
        Utils.save_misc_params(y_axis_param_list, x_axis_list, path_dir, y_label_list)
        
        # save the maximum activation per setting
        #max_activation_list.append(neuron_score_max[0])
        
        with open('max_activations.txt', 'a+') as fd:
            fd.write(str(neuron_score_max[0]) + '\n')
        
        print
        print("Optimisation ends....")
        print('---------------------------')
    
    # Save max activations, one per hyperparameter set
    #Utils.save_max_activation(start_lr_list, max_activation_list, args.output_dir)


if __name__ == "__main__":
    main()
