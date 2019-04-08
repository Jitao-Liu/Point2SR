# -*- coding: UTF-8 -*-
import tensorflow as tf
import tensorflow.contrib.layers as layers


# Fully connected layer
def dense(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, sn=False, with_w=False):

    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "FC"):
        w = tf.get_variable("w", [shape[1], output_size], tf.float32, tf.truncated_normal_initializer(stddev=stddev))
        
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))

        if with_w:
            if sn:
                return tf.matmul(input_, spectral_norm(w)) + bias, spectral_norm(w), bias
            else:
                return tf.matmul(input_, w) + bias, w, bias
        else:
            if sn:
                return tf.matmul(input_, spectral_norm(w)) + bias
            else:
                return tf.matmul(input_, w) + bias


# Convolution layer
def conv2d(input_, output_dim, k_h=1, k_w=1, d_h=1, d_w=1, stddev=0.02, sn=False, scope="conv2d"):
        with tf.variable_scope(scope):
                w = tf.get_variable("w", [k_h, k_w, input_.get_shape()[-1], output_dim], 
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))

                """
                        k_h:Convolution core height
                        k_w:Convolution kernel size
                        input_.get_shape()[-1]:Convolution kernel channel number
                        d_h:Convolution longitudinal step
                        d_w:Convolution lateral step
                        output_dim:Number of convolution kernels
                """
                biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
                if sn:
                    conv = tf.nn.conv2d(input=input_, filter=spectral_norm(w), strides=[1, d_h, d_w, 1],
                                        padding='VALID')
                    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
                else:
                    conv = tf.nn.conv2d(input=input_, filter=w, strides=[1, d_h, d_w, 1], padding='VALID')

                    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def batch_norm(x, is_training=True, scope='batch_normalization'):
    return layers.batch_norm(x, decay=0.9, epsilon=1e-05, center=True, scale=True, updates_collections=None,
                             is_training=is_training, scope=scope)


def instance_norm(x, scope='instance_normalization'):
    return layers.instance_norm(x, epsilon=1e-05, center=True, scale=True, scope=scope)


def layer_norm(x, scope='layer_normalization'):
    return layers.layer_norm(x, center=True, scale=True, scope=scope)


def group_norm(x, scope='group_normalization'):
    return layers.group_norm(x, scope=scope)


def l1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss


def l2_loss(x, y):
    loss = tf.reduce_mean(tf.square(x - y))
    return loss


def generator_loss(fake):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))
    return loss


def discriminator_loss(real, fake):
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))
    loss = real_loss + fake_loss
    return loss


def generator_loss_softplus(fake):
    loss = tf.reduce_mean(tf.nn.softplus(-fake))
    return loss


def discriminator_loss_softplus(real, fake):
    loss = tf.reduce_mean(tf.nn.softplus(fake) + tf.nn.softplus(-real))
    return loss
