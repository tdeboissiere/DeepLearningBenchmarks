import numpy as np
from time import time
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
from keras.datasets import cifar10
import utils
from keras.utils import np_utils


def vgg_block(x, nb_filters, use_bn, nb_conv, bn_axis, data_format):

    for i in range(nb_conv):
        x = Conv2D(filters=nb_filters, kernel_size=(3, 3), padding='same', data_format=data_format)(x)
        if use_bn:
            x = BatchNormalization(scale=False, axis=bn_axis)(x)
        x = Activation("relu")(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', data_format=data_format)(x)

    return x


def run_VGG16(batch_size=16, n_trials=100, use_bn=False, data_format="NCHW"):
    """Run VGG16 experiment

    Args:
        batch_size: mini batch size (default: {16})
        n_trials: number of forward + backward + weight update iterations (default: {100})
    """

    # Determine proper input shape
    if data_format == "NCHW":
        assert K.image_data_format() == 'channels_first', "Change your keras.json file"
        # Update NCHW to channels_first (keras conventions)
        data_format = "channels_first"
        input_shape = (3, 224, 224)
        bn_axis = 1
    else:
        assert K.image_data_format() == 'channels_last', "Change your keras.json file"
        data_format = "channels_last"
        input_shape = (224, 224, 3)
        bn_axis = -1

    img_input = Input(shape=input_shape)
    # Block 1
    x = vgg_block(img_input, 64, use_bn, 2, bn_axis, data_format)
    # Block 2
    x = vgg_block(x, 128, use_bn, 2, bn_axis, data_format)
    # Block 3
    x = vgg_block(x, 256, use_bn, 3, bn_axis, data_format)
    # Block 4
    x = vgg_block(x, 512, use_bn, 3, bn_axis, data_format)
    # Block 5
    x = vgg_block(x, 512, use_bn, 3, bn_axis, data_format)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(units=4096, activation='relu', name='fc1')(x)
    x = Dense(units=4096, activation='relu', name='fc2')(x)
    x = Dense(units=1000, activation='softmax', name='predictions')(x)

    # Create model
    model = Model(inputs=img_input, outputs=x)
    model.summary()

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


def run_SimpleCNN(batch_size):

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, 32, 32)
    else:
        input_shape = (32, 32, 3)

    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv2D(32, 3, 3, border_mode="same", activation="relu")(img_input)
    x = Conv2D(32, 3, 3, border_mode="same", activation="relu")(x)
    x = MaxPooling2D()(x)
    # Block 2
    x = Conv2D(64, 3, 3, border_mode="same", activation="relu")(x)
    x = Conv2D(64, 3, 3, border_mode="same", activation="relu")(x)
    x = MaxPooling2D()(x)
    # Dense part
    x = Flatten(name='flatten')(x)
    x = Dense(512,activation="relu")(x)
    x = Dense(10,activation="relu")(x)

    # Create model
    model = Model(img_input, x)

    opt = SGD()
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    # Input data
    data = cifar10.load_data()
    X, Y = data[0][0], data[0][1]
    Y = np_utils.to_categorical(Y, nb_classes=10)

    # warmup
    model.train_on_batch(X[:32], Y[:32])

    # Split in chunks of size batch size
    num_elem = X.shape[0]
    chunk_size = batch_size
    num_chunks = num_elem / chunk_size
    list_chunks = np.array_split(np.arange(num_elem), num_chunks)

    for e in range(10):
        t0 = time()
        for index, chunk_idx in enumerate(list_chunks):
            X_batch, Y_batch = X[chunk_idx], Y[chunk_idx]
            model.train_on_batch(X_batch, Y_batch)
        t1 = time()

        print t1 - t0

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
