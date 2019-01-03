'''
Created on 8 Nov 2018

@author: Saumitra
'''

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

import Utils
from Utils import LeakyReLU
from Utils import non_local_block
from tensorflow.contrib.rnn import LSTMStateTuple

class ConvGenerator: #TODO Use Resnet Generator to produce 128x80 images, then cut them/zero out the 128-115 timeframes on left and right side. Same with discriminator: Pad all input with 128-115 timeframes
    def __init__(self, model_config):
        self.batch_size = model_config["batch_size"]
        self.num_frames = model_config["num_frames"]
        self.global_noise_dim = model_config["noise_dim"]
        self.local_noise_dim = model_config["rnn_noise_dim_step"]
        self.num_freqs = model_config["num_freqs"]
        self.one_dim_conv = model_config["one_dim_conv"]

    def get_output(self, noise, real_batch_cond, reuse=True):
        with tf.variable_scope("gen", reuse=reuse):
            ch = 128 # 128 for small, 512 for big model
            filter_size = 5

            h = tf.layers.dense(noise, 5*8* ch, activation=LeakyReLU)
            h = tf.reshape(h, (h.shape[0],  5, 8, ch))
            for i in range(4):
                ch /= 2
                #h = self.resnet_block(h, ch)

                if self.one_dim_conv:
                    freqs = 2*h.get_shape().as_list()[1] # Twice the current number since we upsample by factor of 2
                else:
                    freqs = filter_size
                h = tf.layers.conv2d_transpose(h, ch, [freqs, filter_size], (2, 2), "same", activation=LeakyReLU)

            h = tf.layers.conv2d(h, 1, 5, padding="same", activation=None)
            h = tf.maximum(h, tf.log(1e-7))
        target_shape = h.get_shape().as_list()
        target_shape[2] = 115
        h = Utils.crop(h,target_shape)

        tf.summary.scalar("out_min", tf.reduce_min(h), collections=["gen"])
        tf.summary.scalar("out_max", tf.reduce_max(h), collections=["gen"])

        return h

    def resnet_block(self, x, ch):
        upsampled = self.upsample_conv(x, ch)
        return self.residual(upsampled, ch) + upsampled

class RNNGenerator:
    def __init__(self, model_config):
        self.batch_size = model_config["batch_size"]
        self.num_gen_steps = model_config["num_frames"] - model_config["num_cond_frames"]
        self.num_cond_steps = model_config["num_cond_frames"]
        self.global_noise_dim = model_config["noise_dim"]
        self.local_noise_dim = model_config["rnn_noise_dim_step"]
        self.num_output = model_config["num_freqs"]
        self.num_hidden = 512
        self.num_layers = model_config["num_layers"]

    def get_output(self, noise, real_batch_cond, reuse=True):
        ### PREPARE NOISE INPUT
        # Input shape: [batch_size, noise_dim] noise
        assert (noise.get_shape().as_list()[0] == self.batch_size and len(noise.get_shape().as_list()) == 2)

        # Check if we have the right amount of noise: We need batch_size * global_noise_dim + batch_size * num_frames * local_noise_dim in total
        assert (noise.get_shape()[1] == self.global_noise_dim + (self.local_noise_dim * self.num_gen_steps))

        # Split noise input into local and global noise and combine it corrrectly to bring it into correct shape
        # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
        global_noise = noise[:, 0:self.global_noise_dim]
        local_noise = noise[:, self.global_noise_dim:]

        # input = tf.reshape(local_noise, [self.batch_size, self.num_steps, -1]) # Distribute local noise onto every timestep
        input = tf.concat(
            [tf.tile(tf.expand_dims(global_noise, axis=1), multiples=[1, self.num_gen_steps, 1]),
             # Global noise is the same at each input
             tf.reshape(local_noise, [self.batch_size, self.num_gen_steps, -1])], axis=2)  # Local noise

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        input = tf.unstack(input, self.num_gen_steps, 1)

        if self.num_cond_steps == 0:
            out = self.get_unconditional_output(input, reuse)
        else:
            out = self.get_conditional_output(input,real_batch_cond,reuse)
        out = tf.maximum(out, tf.log(1e-7))
        return out

    def get_conditional_output(self, noise, real_batch_cond, reuse):
        #TODO

        raw_rnn_input = tf.transpose(tf.squeeze(real_batch_cond, axis=3), [2,0,1])
        raw_rnn_input_ta = tf.TensorArray(size=self.num_cond_steps, dtype=tf.float32).unstack(raw_rnn_input, 'TBD_Input')
        rnn_cell = rnn.LSTMCell(self.num_hidden, reuse=reuse)

        out_w = tf.get_variable("w", shape=[self.num_hidden, self.num_output], dtype=tf.float32)
        out_b = tf.get_variable("b", shape=[self.num_output], dtype=tf.float32)

        ######## RAW RNN

        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_output is None:
                next_cell_state = rnn_cell.zero_state(self.batch_size, tf.float32)
                next_input = tf.concat([raw_rnn_input_ta.read(0), noise[time]], axis=1)
                next_state = 1
                emit_output = tf.zeros([self.num_output])
            else:  # pass the last state to the next
                next_input = tf.cond(
                    loop_state < self.num_cond_steps,
                    lambda: tf.concat([raw_rnn_input_ta.read(loop_state), noise[time]], axis=1),
                    lambda: tf.concat([tf.matmul(cell_output, out_w) + out_b, noise[time]], axis=1)
                )
                next_state = loop_state + 1
                next_cell_state = cell_state
                # sampling from multinomial
                emit_output = next_input[:,:self.num_output]
            return (time < self.num_cond_steps+self.num_gen_steps, next_input, next_cell_state, emit_output, next_state)

        decoder_emit_ta, _, loop_state_ta = tf.nn.raw_rnn(rnn_cell, loop_fn)
        outputs = decoder_emit_ta.stack()
        outputs = tf.expand_dims(tf.transpose(outputs, [1,2,0]), axis=3)
        return outputs

    def get_unconditional_output(self, noise, reuse):
        with tf.variable_scope("gen", reuse=reuse):
            ####################################

            ### BUILD MODEL
            # Define a lstm cell with tensorflow
            rnn_cell = rnn.MultiRNNCell(
                [rnn.GRUCell(self.num_hidden, reuse=reuse) for _ in range(self.num_layers)])

            # Zero initial state
            initial_state = rnn_cell.zero_state(self.batch_size, tf.float32)

            # Get lstm cell output
            outputs, states = rnn.static_rnn(rnn_cell, noise,
                                             sequence_length=np.repeat(self.num_gen_steps, self.batch_size),
                                             initial_state=initial_state, dtype=tf.float32,
                                             scope="gen_rnn")  # TODO use state or output of context-RNN?

            # DENSE
            dense_out = tf.layers.dense(
                tf.reshape(tf.stack(outputs), [self.num_gen_steps * self.batch_size, -1]), self.num_output,
                activation=LeakyReLU)

            # CONV
            conv = tf.reshape(dense_out, [self.num_gen_steps * self.batch_size, -1, 1, 1])
            conv = tf.layers.conv2d_transpose(conv, 64, [1, 1], activation=LeakyReLU)  # Reverse 1x1 conv on feature map

            #conv = conv[:, :, 0, :]  # [batch, freq, 1, channels] => [batch, freq, channels]
            #for i in range(2):
            #    conv = non_local_block(conv)
            #conv = tf.expand_dims(conv, axis=2)

            # CONV
            conv = tf.layers.conv2d_transpose(conv, 1, [5, 1], activation=None, padding="SAME")
            assert (conv.get_shape().as_list()[2:] == [1, 1])

            # Shared Bias for each frequency bin at the end
            out_biases = tf.get_variable("out_b", shape=[self.num_output], dtype=tf.float32)
            outputs = conv[:,:,0,0] + out_biases
            tf.summary.scalar("out_min", tf.reduce_min(outputs), collections=["gen"])
            tf.summary.scalar("out_max", tf.reduce_max(outputs), collections=["gen"])
            outputs = tf.reshape(outputs, [self.num_gen_steps, self.batch_size, self.num_output, 1])
            outputs = tf.transpose(outputs, [1, 2, 0, 3])

            assert (outputs.get_shape().as_list() == [self.batch_size, self.num_output, self.num_gen_steps, 1])

            return outputs

    def conv_output(self, rnn_output_batch):
        dense_out = tf.layers.dense(rnn_output_batch, self.num_output, activation=LeakyReLU)
        conv = tf.reshape(dense_out, [self.batch_size, -1, 1, 1])
        conv = tf.layers.conv2d_transpose(conv, 64, [1, 1], activation=LeakyReLU)  # Reverse 1x1 conv on feature map

        conv = conv[:, :, 0, :]  # [batch, freq, 1, channels] => [batch, freq, channels]
        for i in range(2):
            conv = non_local_block(conv)
        conv = tf.expand_dims(conv, axis=2)

        conv = tf.layers.conv2d_transpose(conv, 1, [5, 1], activation=None, padding="SAME")
        assert (conv.get_shape().as_list()[2:] == [1, 1])

        # Shared Bias for each frequency bin at the end
        out_biases = tf.get_variable("out_b", shape=[self.num_output], dtype=tf.float32)
        outputs = tf.nn.relu(conv[:, :, 0,
                             0] + out_biases)  # TODO maybe some sort of activation function here depending on the range of values our samples can have
        return outputs