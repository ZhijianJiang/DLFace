#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Zhijian Jiang"
__email__ = "zhijian.jiang@columbia.edu"

"""
VGGFace models using keras

Reference: 
* https://github.com/rcmalli/keras-vggface/blob/master/keras_vggface/models.py
* HW 5 on COMS 4995 Deep Learning for Computer Vision, Spring 2018
"""

from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Reshape, Permute, multiply, ZeroPadding2D
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.engine.topology import get_source_inputs
import warnings
from keras.models import Sequential
from keras import layers


def build_model(identities=1680, input_shape=(125, 94, 3)):
    """
    Args:
        identities (int, optional): Default to 1680. Number of identities.
        input_shape (tuple of int, optional): Default to (125, 94, 1). Size of image.

    Returns:
        model
    """
    model = Sequential()

    # vgg part
    if K.backend() == 'tensorflow':
        print('The current backend is tensorflow')

    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # top part
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dense(identities, activation='softmax', name='fc8'))

    return model


def main():
    """
    for test
    """
    model = build_model()
    print(model.summary())


if __name__ == '__main__':
    main()
