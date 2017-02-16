import os
import numpy as np
from time import time
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD


def run_VGG16(batch_size=16, n_trials=100):
    """Run VGG16 experiment

    Args:
        batch_size: mini batch size (default: {16})
        n_trials: number of forward + backward + weight update iterations (default: {100})
    """

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, 224, 224)
    else:
        input_shape = (224, 224, 3)

    img_input = Input(shape=input_shape)
    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

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
    print(os.path.basename(__file__))
    print("Keras version: %s" % keras.__version__)
    print("Keras backend: %s" % K.backend())
    print("Backend version: %s" % backend.__version__)
    print("Mean Time per update: %7.3f ms." % (1000. * (t1 - t0) / n_trials))
