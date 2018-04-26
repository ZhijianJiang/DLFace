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

# yq2230
"""This part trains the vgg network"""
tensorboard_callback = TensorBoard(log_dir='./logs/largedata', histogram_freq=0,
                                   write_graph=True, write_images=False)
checkpoint = ModelCheckpoint("./large_model.h5", monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min')


def build_vgg16(framework='tf'):
    if framework == 'th':
        # build the VGG16 network in Theano weight ordering mode
        backend.set_image_dim_ordering('th')
    else:
        # build the VGG16 network in Tensorflow weight ordering mode
        backend.set_image_dim_ordering('tf')

    model = Sequential()
    if framework == 'th':
        model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))
    else:
        model.add(ZeroPadding2D((1, 1), input_shape=(img_width, img_height, 3)))

    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))
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

    return model


# yq2230
""" This part loads the vgg weights """

weights_path = './data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
tf_model = build_vgg16('tf')
tf_model.load_weights(weights_path)

# yq
"""" build a classifier model to put on top of the convolutional model """
top_model = Sequential()
top_model.add(Flatten(input_shape=tf_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(nc, activation='sigmoid'))

# yq2230
# add the model on top of the convolutional base
tf_model.add(top_model)
tf_model.summary()

from keras.models import load_model

tf_model = load_model('./large_model.h5')

for layer in tf_model.layers[:25]:
    layer.trainable = False

tf_model.compile(loss='categorical_crossentropy',
                 optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                 metrics=['accuracy'])

tf_model.fit_generator(train_generator,
                       initial_epoch=0,
                       verbose=1,
                       validation_data=validation_generator,
                       steps_per_epoch=steps_per_epoch_train,
                       epochs=1,
                       callbacks=[tensorboard_callback, checkpoint])
# yq2230
# Before unfreezing the first 25 layers, we got a validation accuracy of 83%.


tf_model = load_model('./large_model2.h5')
checkpoint = ModelCheckpoint("./large_model2.h5", monitor='val_loss',
                             verbose=1, save_best_only=True, mode='min')

for layer in tf_model.layers[:25]:
    layer.trainable = True

tf_model.compile(loss='categorical_crossentropy',
                 optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                 metrics=['accuracy'])

tf_model.fit_generator(train_generator,
                       initial_epoch=0,
                       verbose=1,
                       validation_data=validation_generator,
                       steps_per_epoch=steps_per_epoch_train,
                       epochs=50,
                       callbacks=[tensorboard_callback, checkpoint])

# yq2230
# After unfreezing and training the whole network, the accuracy finnaly reaches 87%.


# yq2230
eva_data_dir = './data/c_test'

eva_datagen = ImageDataGenerator(rescale=1. / 255)

eva_generator = eva_datagen.flow_from_directory(
    eva_data_dir,
    target_size=(img_width, img_height),
    batch_size=1)

tf_model.evaluate_generator(eva_generator)

# The accuracy on the test set is 89%. We have successfully built a easy baseline model for face recognition.

# yq2230
# Then we load the trained model, and we need to remove the last two layers, for applying CAM
tf_model = load_model('./large_model2.h5')
tf_model.pop()
tf_model.pop()

# yq2230
# Add a global average pooling layer, a dense layer and a softmax output layer
from keras.layers.core import Lambda
from keras import backend as K
from keras.optimizers import SGD

def global_average_pooling(x):
    return K.mean(x, axis = (2, 3))

def global_average_pooling_shape(input_shape):
    return input_shape[0:2]

tf_model.add(Lambda(global_average_pooling, output_shape=global_average_pooling_shape))
tf_model.add(Dense(nc, activation = 'softmax', init='uniform'))
sgd = SGD(lr=0.0000001, decay=1e-6, momentum=0.5, nesterov=True)
tf_model.summary()

checkpoint = ModelCheckpoint("./large_model2_cam.h5", monitor='val_loss', 
                             verbose=1, save_best_only=True, mode='min')

#yq2230
#Train the model 
tf_model.compile(loss = 'categorical_crossentropy', \
optimizer = sgd, metrics=['accuracy'])

tf_model.fit_generator(train_generator, 
              initial_epoch=0, 
              verbose=1, 
              validation_data=validation_generator, 
              steps_per_epoch=steps_per_epoch_train, 
              epochs=50, 
              callbacks=[tensorboard_callback, checkpoint])

# yq2230
# function to do CAM
import cv2
from array import array

def visualize_class_activation_map(model_path, img_path, output_path):
    model = load_model(model_path)
    original_img = cv2.imread(img_path, 1)
#     print(original_img)
    width, height, _ = original_img.shape
#     print(width)
#     print(height)

    img = np.array([np.transpose(np.float32(original_img), (0, 1, 2))])
    print(img.shape)

    class_weights = model.layers[-1].get_weights()[0]
#     print(class_weights)
    final_conv_layer = get_output_layer(model, "conv5_3")
    get_output = K.function([model.layers[0].input], \
                [final_conv_layer.output, 
    model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img])
    conv_outputs = conv_outputs[0, :, :, :]

    #Create the class activation map.
    cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[1:3])
    target_class = 0
    for i, w in enumerate(class_weights[:, target_class]):
            cam += w * conv_outputs[i, :, :]
#     print "predictions", predictions
    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    img = heatmap*0.5 + original_img
    cv2.imwrite(output_path, img)

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

# yq2230
# take out several pics to try CAM
visualize_class_activation_map("./large_model2_cam.h5", './data/c_validation/Alexa_Vega/17_Alexa_Vega_0003.jpg', "heatmap_alexa_vega.jpg")




# ky2371
""" Here we pick some of the images that we make mistake to have a look """

errors = []

for X_batch, Y_batch in eva_generator:
    expected = np.argmax(Y_batch, axis=1)
    print(expected)

    if expected.item((0)) == 100:
        prediction = np.argmax(tf_model.predict(X_batch), axis=1)

        res = expected - prediction
        for index in np.flatnonzero(res):
            errors.append((X_batch[index], expected[index], prediction[index]))

# ky2371
"""This function will map the class index into the person's name"""


def find_class_name(index, indices):
    for name in indices:
        if indices[name] == index:
            return name

    return None


# ky2371
"""This part will show the images that we make mistakes"""
for x, y, p in errors:
    plt.imshow(x)
    plt.title("Class = %s, Predict = %s" % (find_class_name(y, eva_generator.class_indices),
                                            find_class_name(p, eva_generator.class_indices)))
    plt.show()

# ky2371
""" Now we will try to use our model to see if two photos belong to the same person.
This function calculate the cosine similarity between the output of the network using
two different images as the input. """


def get_similarity(a, b, model):
    image0 = np.asarray(Image.open(a)) / 255
    image1 = np.asarray(Image.open(b)) / 255
    image0 = np.expand_dims(image0, axis=0)
    image1 = np.expand_dims(image1, axis=0)
    print(np.argmax(model.predict(image0)))
    print(np.argmax(model.predict(image1)))
    return 1 - spatial.distance.cosine(model.predict(image0),
                                       model.predict(image1))


# ky2371
""" Here we use the verification set to see if our network can verify a person"""
import os
from PIL import Image

files = os.listdir("./data/CACD_ver/CACD_VS")
right_res = []

for i in range(2000):
    prefix = "./data/CACD_ver/CACD_VS/"

    fold = i // 200
    index = i % 200
    match = fold * 400 + index
    mismatch = match + 200

    image0 = "%s%04d_0.jpg" % (prefix, match)
    image1 = "%s%04d_1.jpg" % (prefix, match)
    image2 = "%s%04d_0.jpg" % (prefix, mismatch)
    image3 = "%s%04d_1.jpg" % (prefix, mismatch)

    sim_match = get_similarity(image0, image1, tf_model)
    sim_mismatch = get_similarity(image2, image3, tf_model)

    if sim_match > 0.5:
        right_res.append((image0, image1, image2, image3, match, mismatch, sim_match, sim_mismatch))
        print(i)
        print("%4f %4f" % (sim_match, sim_mismatch))

for image0, image1, image2, image3, match, mismatch, sim_match, sim_mismatch in right_res:
    #     if sim_mismatch > 0.5:
    print(sim_mismatch)
    print(mismatch)
    plt.imshow(np.asarray(Image.open(image0)))
    plt.show()
    plt.imshow(np.asarray(Image.open(image1)))
    plt.show()
