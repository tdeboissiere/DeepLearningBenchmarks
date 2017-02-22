import numpy as np
from time import time
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
import utils


def vgg_block(x, nb_filters, use_bn, nb_conv, bn_axis, bn_mode):

    for i in range(nb_conv):
        x = Convolution2D(nb_filters, 3, 3, border_mode='same')(x)
        if use_bn:
            x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
        x = Activation("relu")(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    return x


def run_VGG16(batch_size=16, n_trials=100, use_bn=False, bn_mode=2):
    """Run VGG16 experiment

    Args:
        batch_size: mini batch size (default: {16})
        n_trials: number of forward + backward + weight update iterations (default: {100})
    """

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, 224, 224)
        bn_axis = 1
    else:
        input_shape = (224, 224, 3)
        bn_axis = -1

    img_input = Input(shape=input_shape)
    # Block 1
    x = vgg_block(img_input, 64, use_bn, 2, bn_axis, bn_mode)
    # Block 2
    x = vgg_block(x, 128, use_bn, 2, bn_axis, bn_mode)
    # Block 3
    x = vgg_block(x, 256, use_bn, 3, bn_axis, bn_mode)
    # Block 4
    x = vgg_block(x, 512, use_bn, 3, bn_axis, bn_mode)
    # Block 5
    x = vgg_block(x, 512, use_bn, 3, bn_axis, bn_mode)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1000, activation='softmax', name='predictions')(x)

    # Create model
    model = Model(img_input, x)

    opt = SGD()
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    # Input data
    X = np.zeros((batch_size,) + input_shape, dtype=np.float32)
    Y = np.zeros((batch_size, 1000), dtype=np.float32)

    # warmup
    model.train_on_batch(X, Y)

    t0 = time()
    for i in range(n_trials):
        model.train_on_batch(X, Y)
    t1 = time()

    # Import backend to get version number
    if K.backend() == "tensorflow":
        import tensorflow as backend
    elif K.backend() == "theano":
        import theano as backend

    # Print summary
    utils.print_module("Keras version: %s" % keras.__version__)
    utils.print_module("Keras backend: %s" % K.backend())
    utils.print_module("Backend version: %s" % backend.__version__)
    utils.print_result("%7.3f ms." % (1000. * (t1 - t0) / n_trials))
