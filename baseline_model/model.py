import os
import h5py
import matplotlib.pyplot as plt
import time, pickle, pandas
from scipy import spatial
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend
from keras import optimizers

# ky2371
""" This part reads the dataset from directories that are classified by the shell script """
""" The dataset is divided into three parts, training, validation and test with ratio of 8:1:1 """
batch_size = 32
img_width, img_height = 250, 250
train_data_dir = './data/c_train'
validation_data_dir = './data/c_validation'

train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size)

nb_train_samples = 14525
nb_validation_samples = 1867

steps_per_epoch_train = nb_train_samples / batch_size
steps_per_epoch_val = nb_validation_samples / batch_size

nc = len(train_generator.class_indices)

# ky2371
""" Now we try to train a deep model by ourself instead of using VGG, 
which is proposed in Learning Face Representation from Scratch. """


def build_network():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(img_width, img_height, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', name='conv1_11'))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_12'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_21'))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv1_22'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(96, (3, 3), activation='relu', name='conv1_31'))
    model.add(Conv2D(192, (3, 3), activation='relu', name='conv1_32'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv1_41'))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv1_42'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(160, (3, 3), activation='relu', name='conv1_51'))
    model.add(Conv2D(320, (3, 3), name='conv1_52'))
    model.add(AveragePooling2D((7, 7), strides=(1, 1)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(nc, activation='softmax'))

    return model


# ky2371
""" Here we train the network """
model = build_network()

# model = load_model('./new_model.h5')
tensorboard_callback = TensorBoard(log_dir='./logs/new_modellogs',
                                   histogram_freq=0, write_graph=True, write_images=False)
checkpoint = ModelCheckpoint("./new_model.h5", monitor='val_loss',
                             verbose=1, save_best_only=True, mode='min')

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=1e-6),
              metrics=['accuracy'])

model.fit_generator(train_generator,
                    initial_epoch=0,
                    verbose=1,
                    validation_data=validation_generator,
                    steps_per_epoch=steps_per_epoch_train,
                    epochs=50,
                    callbacks=[tensorboard_callback, checkpoint])

# ky2371
""" Here will evaluate our network on the test set """

eva_data_dir = './data/c_test'

eva_datagen = ImageDataGenerator(rescale=1. / 255)

eva_generator = eva_datagen.flow_from_directory(
    eva_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size)

model.evaluate_generator(eva_generator)
