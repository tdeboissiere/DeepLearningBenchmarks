import os
import time
import numpy as np
import tensorflow as tf
import utils


def conv2d(x, n_in, n_out, k, s, p='SAME', bias=True, data_format="NCHW", scope=None):
    with tf.variable_scope(scope or 'Conv2D'):
        kernel_init_std = np.sqrt(2.0 / (k * k * n_in))
        kernel = tf.get_variable('Weight', shape=[k,k,n_in,n_out],
                                 initializer=tf.truncated_normal_initializer(0.0, kernel_init_std))
        tf.add_to_collection('Weights', kernel)
        y = tf.nn.conv2d(x, kernel, [1,1,s,s], padding=p, data_format=data_format)
        if bias is True:
            bias = tf.get_variable('Bias', shape=[n_out],
                                   initializer=tf.constant_initializer(0.0))
            tf.add_to_collection('Biases', bias)
            y = tf.nn.bias_add(y, bias, data_format=data_format)
    return y


def linear(x, n_in, n_out, bias=True, scope=None):
    with tf.variable_scope(scope or 'Linear'):
        weight_init_std = np.sqrt(1.0 / n_out)
        weight = tf.get_variable('Weight', shape=[n_in,n_out],
                                 initializer=tf.truncated_normal_initializer(0.0, weight_init_std))
        tf.add_to_collection('Weights', weight)
        y = tf.matmul(x, weight)
        if bias is True:
            bias = tf.get_variable('Bias', shape=[n_out],
                                   initializer=tf.constant_initializer(0.0))
            tf.add_to_collection('Biases', bias)
            y = y + bias
    return y


class Vgg16Model():
    """ VGG16 model adapted from https://github.com/machrisaa/tensorflow-vgg"""
    def __init__(self, data_format="NCHW", use_bn=False, use_fused=False):
        self.image_mean = np.array([103.939, 116.779, 123.68])
        self.data_format = data_format
        self.use_bn = use_bn
        self.use_fused = use_fused
        if self.data_format == "NCHW":
            self.pooling_order = [1,1,2,2]
        elif self.data_format == "NHWC":
            self.pooling_order = [1,2,2,1]

    def _vgg_conv_relu(self, x, n_in, n_out, scope):
        with tf.variable_scope(scope):
            conv = conv2d(x, n_in, n_out, 3, 1, p='SAME', data_format=self.data_format)
            if self.use_bn:
                conv = tf.contrib.layers.batch_norm(conv, data_format=self.data_format, fused=self.use_fused)
            relu = tf.nn.relu(conv)
        return relu

    def _vgg_max_pool(self, x, scope):
        with tf.variable_scope(scope):
            pool = tf.nn.max_pool(x, self.pooling_order, self.pooling_order,
                                  padding='SAME', data_format=self.data_format)
        return pool

    def _vgg_fully_connected(self, x, n_in, n_out, scope):
        with tf.variable_scope(scope):
            fc = linear(x, n_in, n_out)
        return fc

    def __call__(self, x, scope=None):
        with tf.variable_scope(scope or 'Vgg16'):
            # conv stage 1
            relu1_1 = self._vgg_conv_relu(x, 3, 64, 'conv1_1')
            relu1_2 = self._vgg_conv_relu(relu1_1, 64, 64, 'conv1_2')
            pool1 = self._vgg_max_pool(relu1_2, 'pool1')
            # conv stage 2
            relu2_1 = self._vgg_conv_relu(pool1, 64, 128, 'conv2_1')
            relu2_2 = self._vgg_conv_relu(relu2_1, 128, 128, 'conv2_2')
            pool2 = self._vgg_max_pool(relu2_2, 'pool2')
            # conv stage 3
            relu3_1 = self._vgg_conv_relu(pool2, 128, 256, 'conv3_1')
            relu3_2 = self._vgg_conv_relu(relu3_1, 256, 256, 'conv3_2')
            relu3_3 = self._vgg_conv_relu(relu3_2, 256, 256, 'conv3_3')
            pool3 = self._vgg_max_pool(relu3_3, 'pool3')
            # conv stage 4
            relu4_1 = self._vgg_conv_relu(pool3, 256, 512, 'conv4_1')
            relu4_2 = self._vgg_conv_relu(relu4_1, 512, 512, 'conv4_2')
            relu4_3 = self._vgg_conv_relu(relu4_2, 512, 512, 'conv4_3')
            pool4 = self._vgg_max_pool(relu4_3, 'pool4')
            # conv stage 5
            relu5_1 = self._vgg_conv_relu(pool4, 512, 512, 'conv5_1')
            relu5_2 = self._vgg_conv_relu(relu5_1, 512, 512, 'conv5_2')
            relu5_3 = self._vgg_conv_relu(relu5_2, 512, 512, 'conv5_3')
            pool5 = self._vgg_max_pool(relu5_3, 'pool5')
            # fc6
            n_conv_out = 7 * 7 * 512
            flatten = tf.reshape(pool5, [-1, n_conv_out])
            fc6 = self._vgg_fully_connected(flatten, n_conv_out, 4096, scope='fc6')
            relu_6 = tf.nn.relu(fc6)
            # fc7
            fc7 = self._vgg_fully_connected(relu_6, 4096, 4096, scope='fc7')
            relu_7 = tf.nn.relu(fc7)
            # fc8, prob
            fc8 = self._vgg_fully_connected(relu_7, 4096, 1000, scope='fc8')
            prob = tf.nn.softmax(fc8)
            return prob


def run_VGG16(batch_size, n_trials, data_format="NHWC", use_XLA=False, use_bn=False, use_fused=False):
    """Run VGG16 experiments in pure tensorflow

    Args:
        batch_size: mini batch size
        n_trials: number of forward + backward + weight update trials
        data_format: image dimension ordering (default: {"NHWC"})
        use_XLA: if True, use XLA compiler (default: {False})
        use_bn: if True, use BatchNorm in conv layers (default: {False})
        use_XLA: if True, use Fused BatchNorm in conv layers (default: {False})
    """

    with tf.Graph().as_default(), tf.device('/gpu:0'):

        if data_format == "NHWC":
            input_shape = (batch_size, 224, 224, 3)
        elif data_format == "NCHW":
            input_shape = (batch_size, 3, 224, 224)

        # Initialize inputs
        train_inputs = tf.random_uniform(input_shape)
        # Initialize target
        labels = tf.one_hot(np.arange(batch_size), on_value=1.0, off_value=0.0, depth=1000)

        vgg16 = Vgg16Model(data_format=data_format, use_bn=use_bn, use_fused=use_fused)
        predictions = vgg16(train_inputs, scope='Vgg16')

        # Loss function
        loss = tf.losses.softmax_cross_entropy(labels, predictions)
        # Optimizer
        opt = tf.train.GradientDescentOptimizer(learning_rate=1E-1)
        # Calculate the gradients for the batch of data
        grads = opt.compute_gradients(loss)
        # Weight update op
        apply_gradient_op = opt.apply_gradients(grads)

        if use_XLA:
            config = tf.ConfigProto()
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        else:
            config = None

        # Run a session
        with tf.Session(config=config) as sess:

            # Initialize variables
            init = tf.global_variables_initializer()
            sess.run(init)

            # warmup run
            sess.run([apply_gradient_op])

            t0 = time.time()
            for i in range(n_trials):
                sess.run([apply_gradient_op])
            t1 = time.time()

    # Print summary
    utils.print_module("tensorflow version: %s" % tf.__version__)
    utils.print_result("%7.3f ms." % (1000. * (t1 - t0) / n_trials))
