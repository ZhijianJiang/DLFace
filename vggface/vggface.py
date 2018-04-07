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


# load data
lfw_loader = LFWLoader()
X_train, X_test, y_train, y_test = lfw_loader.load_lfw_data()


# store class name to a dict
nb_class_names = len(lfw_loader.target_names)
class_names = {}
for i in range(nb_class_names):
    class_names[i] = lfw_loader.target_names[i]


# various sizes
img_width = X_train.shape[1]
img_height = X_train.shape[2]
nb_train_samples = X_train.shape[0]
nb_test_samples = X_test.shape[0]
batch_size = 32
steps_per_epoch_train = nb_train_samples / batch_size
steps_per_epoch_test = nb_test_samples / batch_size

nb_classes = 1680

# build the model
model = build_model()
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# data generator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_datagen.fit(X_train)
test_datagen.fit(X_test)

train_datagenerator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
test_datagenerator = test_datagen.flow(X_test, y_test, batch_size=batch_size)

# callback
tensorboard_callback = TensorBoard(log_dir='./logs/', histogram_freq=0,
                                   write_graph=True, write_images=False)
checkpoint_callback = ModelCheckpoint('./models/vgg.h5', monitor='val_acc', verbose=0, save_best_only=True,
                                      save_weights_only=False, mode='auto', period=1)

model.fit_generator(train_datagenerator, steps_per_epoch=steps_per_epoch_train, epochs=10,
                    callbacks=[tensorboard_callback, checkpoint_callback], validation_data=test_datagenerator,
                    verbose=1)


