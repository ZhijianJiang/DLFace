#!/usr/bin/env python

from models import *
import sys


# zj2226
def main():
    """
    This is the main function
    """

    # load the dataset
    nb_train = 14513
    nb_test = 1791
    img_width, img_height = 250, 250
    train_data_dir = sys.argv[1]
    validation_data_dir = sys.argv[2]

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=nb_train)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=nb_test)

    X_train, y_train = train_generator.next()
    X_test, y_test = test_generator.next()

    # build the model and load pretrained vgg16 weights
    # weights_path = '/imagenet_vgg16_fine-tuning/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights_path = sys.argv[3]
    tf_model = build_vgg16('tf', img_width=250, img_height=250)
    tf_model.load_weights(weights_path)

    # build a classifier model to put on top of the convolutional model
    tf_model.add(build_resnet(input_shape=(7, 7, 512)))

    # train the network
    tensorboard_callback = TensorBoard(log_dir='./logs/cacd2000_dropout', histogram_freq=0, write_graph=True,
                                       write_images=False)
    checkpoint = ModelCheckpoint("./models/cacd2000_dropout.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 mode='min')
    import math
    lr = 1e-3
    past_loss = math.inf
    while lr > 1e-12:
        print('Current learning rate = ' + str(lr))
        tf_model.compile(loss='categorical_crossentropy',
                         optimizer=optimizers.SGD(lr=lr, momentum=0.9),
                         metrics=['accuracy'])
        tf_model.fit(x=X_train,
                     y=y_train,
                     batch_size=16,
                     epochs=1,
                     verbose=1,
                     validation_data=(X_test, y_test),
                     callbacks=[tensorboard_callback, checkpoint],
                     shuffle=True)
        loss, acc = tf_model.evaluate(X_test, y_test, verbose=0)
        print('past loss = ' + str(past_loss) + ' loss = ' + str(loss))
        if past_loss < loss:
            lr = lr * 0.1  # update lr if loss not improve
        else:
            past_loss = loss  # update loss if loss improve

# zj2226
if __name__ == '__main__':
    """
    This is to call the main function
    """
    main()