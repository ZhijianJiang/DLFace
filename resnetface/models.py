from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D, Input
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization, Activation
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend
from keras import optimizers
from keras import layers


# zj2226
def res_block(input_tensor, kernel_size, filters, strides=(1, 1)):
    """
    This is the fuConstruct a residual block

                        |------------------------>
    kernel_size x kernel_size conv, filter1      |
                        |                        |
    kernel_size x kernel_size conv, filter2      |
                        |                        |
    1 x 1 conv, filter2                          |
                        |<-----------------------                         |

    Args:
        input_tensor (Tensor): input data
        kernel_size (int): size of 2nd layers of conv2d
        filters (tuple of int): 1 * 3 filter value of 1st conv and 2nd conv

    Reference:
        https://blog.waya.ai/deep-residual-learning-9610bb62c355
        https://blog.csdn.net/wspba/article/details/56019373
        https://blog.csdn.net/qq_25491201/article/details/78405549
    """
    #     shortcut = input_tensor
    filter1, filter2 = filters

    x = Conv2D(filter1, kernel_size=kernel_size, strides=strides, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filter2, kernel_size=(1, 1), strides=strides, padding='same')(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


#zj2226
def build_resnet(framework='tf', nb_res=(3, 0, 0, 0), input_shape=(3, 250, 250), nc=200, dropout=False):
    """
    This is the function to build the resnet
    """

    if framework == 'th':
        # build the VGG16 network in Theano weight ordering mode
        backend.set_image_dim_ordering('th')
    else:
        # build the VGG16 network in Tensorflow weight ordering mode
        backend.set_image_dim_ordering('tf')

    input_tensor = Input(shape=input_shape)
    x = input_tensor

    # residual units

    for i in range(nb_res[0]):
        x = res_block(x, 3, (64, 64))

    for i in range(nb_res[1]):
        x = res_block(x, 3, (128, 128))

    for i in range(nb_res[2]):
        x = res_block(x, 3, (256, 256))

    for i in range(nb_res[3]):
        x = res_block(x, 3, (512, 512))

    x = layers.GlobalAveragePooling2D()(x)

    if dropout:
        x = layers.Dropout(0.1)(x)

    x = Dense(units=nc, kernel_initializer="he_normal",
              activation="softmax")(x)

    model = Model(input_tensor, x)
    return model


# Prof. Peter
def build_vgg16(framework='tf', img_width=250, img_height=250):
    """
    This is function to build VGG-16
    """
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