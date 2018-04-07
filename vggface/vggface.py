#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Zhijian Jiang"
__email__ = "zhijian.jiang@columbia.edu"

"""
Train VGGFace models

Reference: HW 5 in DLCV
"""

from LFW import LFWLoader
from models import build_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint
from numpy import argmax

class VGGFace:
    def __init__(self):
        """
        Init the LFW loader and build the model
        """
        self.lfw_loader = LFWLoader()

        # build the model
        self.model = build_model()

        # store class name to a dict
        nb_class_names = len(self.lfw_loader.target_names)
        self.class_names = {}
        for i in range(nb_class_names):
            self.class_names[i] = self.lfw_loader.target_names[i]

    def fit(self, X_train, y_train, X_test, y_test, batch_size=32):
        """
        Train the model

        Args:
            X_train:
            y_train:
            batch_size:

        Returns:
        """
        self.model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

        # various sizes
        img_width = X_train.shape[1]
        img_height = X_train.shape[2]
        nb_train_samples = X_train.shape[0]
        batch_size = 32
        steps_per_epoch_train = nb_train_samples / batch_size

        nb_classes = 1680

        # data generator
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_datagen.fit(X_train)
        test_datagen.fit(X_test)

        train_datagenerator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
        test_datagenerator = test_datagen.flow(X_test, y_test, batch_size=batch_size)

        # callback
        tensorboard_callback = TensorBoard(log_dir='./logs/', histogram_freq=0,
                                           write_graph=True, write_images=False)
        checkpoint_callback = ModelCheckpoint('./models/vgg.h5', monitor='val_acc', verbose=0, save_best_only=True,
                                              save_weights_only=False, mode='auto', period=1)

        self.model.fit_generator(train_datagenerator, steps_per_epoch=steps_per_epoch_train, epochs=10,
                            callbacks=[tensorboard_callback, checkpoint_callback], validation_data=test_datagenerator,
                            verbose=1)

    def evaluate(self, X_test, y_test):
        # nb_test_samples = X_test.shape[0]
        # steps_per_epoch_test = nb_test_samples / batch_size
        pass

    def predict(self, X, batch_size=1, model_path=''):
        """
        Predict the person of this face

        Args:
            X: input image
            batch_size
            model_path:

        Returns:
        """
        if model_path != '':
            self.model.load_weights(model_path)
        return self.class_names[argmax(self.model.predict(X, batch_size=batch_size), axis=1)]


def main():
    vggface = VGGFace()

    # load data
    X_train, X_test, y_train, y_test = vggface.lfw_loader.load_lfw_data()
    vggface.fit(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()