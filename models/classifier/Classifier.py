'''
Created on 13 Nov 2018

@author: Saumitra
'''
import tensorflow as tf

def architecture(input_var, train_mode):
    
    print_layer_shapes = False
    model = {}
    #print(train_mode)
    
    # Conv1
    model['conv1'] = tf.layers.conv2d(inputs = input_var, filters = 64, kernel_size = (3, 3), kernel_initializer = tf.orthogonal_initializer)
    model['act_out_conv1'] = tf.nn.leaky_relu(features=model['conv1'], alpha=0.01)
    if print_layer_shapes:
        print(model['conv1'].shape)
        print(model['act_out_conv1'].shape)
    
    # Conv2
    model['conv2'] = tf.layers.conv2d(inputs = model['act_out_conv1'], filters = 32, kernel_size = (3, 3), kernel_initializer = tf.orthogonal_initializer)
    model['act_out_conv2'] = tf.nn.leaky_relu(features=model['conv2'], alpha=0.01)
    if print_layer_shapes:    
        print(model['conv2'].shape)
        print(model['act_out_conv2'].shape)
    
    # Maxpool3
    model['pool3'] = tf.layers.max_pooling2d(inputs=model['act_out_conv2'], pool_size=[3, 3], strides=(3, 3))
    if print_layer_shapes:    
        print(model['pool3'].shape)
    
    # Conv4
    model['conv4'] = tf.layers.conv2d(inputs = model['pool3'], filters = 128, kernel_size = (3, 3), kernel_initializer = tf.orthogonal_initializer)
    model['act_out_conv4'] = tf.nn.leaky_relu(features=model['conv4'], alpha=0.01)
    if print_layer_shapes:
        print(model['conv4'].shape)
        print(model['act_out_conv4'].shape)
    
    # Conv5
    model['conv5'] = tf.layers.conv2d(inputs = model['act_out_conv4'], filters = 64, kernel_size = (3, 3), kernel_initializer = tf.orthogonal_initializer)
    model['act_out_conv5'] = tf.nn.leaky_relu(features=model['conv5'], alpha=0.01)
    if print_layer_shapes:
        print(model['conv5'].shape)
        print(model['act_out_conv5'].shape)
    
    # Maxpool6
    model['pool6'] = tf.layers.max_pooling2d(inputs=model['act_out_conv5'], pool_size=[3, 3], strides=(3, 3))
    if print_layer_shapes:
        print(model['pool6'].shape)
    
    # Flatten
    model['pool6_flat'] = tf.layers.flatten(model['pool6'])
    if print_layer_shapes:
        print(model['pool6_flat'].shape)
    
    # FC7
    model['fc7'] = tf.layers.dense(inputs= tf.layers.dropout(model['pool6_flat'], rate=0.5, training = train_mode), units=256, kernel_initializer = tf.orthogonal_initializer)
    model['act_out_fc7'] = tf.nn.leaky_relu(features=model['fc7'], alpha = 0.01)
    if print_layer_shapes:
        print(model['fc7'].shape)
        print(model['act_out_fc7'].shape)
    
    # FC8
    model['fc8'] = tf.layers.dense(inputs= tf.layers.dropout(model['act_out_fc7'], rate=0.5, training = train_mode), units=64, kernel_initializer = tf.orthogonal_initializer)
    model['act_out_fc8'] = tf.nn.leaky_relu(features=model['fc8'], alpha = 0.01)
    if print_layer_shapes:
        print(model['fc8'].shape)
        print(model['act_out_fc8'].shape)    
    
    model['fc9'] = tf.layers.dense(inputs= tf.layers.dropout(model['act_out_fc8'], rate=0.5, training = train_mode), units=1, activation=tf.nn.sigmoid, kernel_initializer = tf.orthogonal_initializer)
    if print_layer_shapes:
        print(model['fc9'].shape)

    return model

