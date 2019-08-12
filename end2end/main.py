# -*- coding:utf-8 -*-
__author__ = 'shichao'

import argparse
from datetime import datetime
import os
from keras_applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, warnings
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.utils import get_file
from keras.utils import layer_utils
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
import os

def get_network(args):
    if args.net == 'squeezenet':
        from squeezenet import SqueezeNet
        net = SqueezeNet
    else:
        raise RuntimeError('the network is not supported yet')
    return net

def main():


    image_size1 = 640
    image_size2 = 480

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    # parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    # parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    # parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    # parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-path', type=str, default='./', help='root data path')
    parser.add_argument('-e',type=int,default=20,help='epoch')
    args = parser.parse_args()

    root = args.path
    train_dir = os.path.join(root,'train')
    validation_dir = os.path.join(root,'val')
    num_classes = len(os.listdir(train_dir))
    epoch = args.e

    net = get_network(args)
    conv = net(classes=num_classes)
    model = models.Sequential()
    model.add(conv)

    model.add(GlobalMaxPooling2D())
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_batchsize = args.b
    val_batchsize = args.b
    # Data Generator for Training data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size1, image_size2),
        batch_size=train_batchsize,
        class_mode='categorical')

    # Data Generator for Validation data
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size1, image_size2),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    # Train the Model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples / train_generator.batch_size,
        epochs=epoch,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples / validation_generator.batch_size,
        verbose=1)

    # Save the Model
    model.save('squeeze_{0}_classes_nhwc_epoch{1}.h5'.format(num_classes, epoch))

    # Plot the accuracy and loss curves
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    Validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size1, image_size2),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

    # Get the filenames from the generator
    fnames = validation_generator.filenames

    # Get the ground truth from generator
    ground_truth = validation_generator.classes

    # Get the label to class mapping from the generator
    label2index = validation_generator.class_indices

    # Getting the mapping from class index to class label
    idx2label = dict((v, k) for k, v in label2index.items())

    # Get the predictions from the model using the generator
    predictions = model.predict_generator(validation_generator,
                                          steps=validation_generator.samples / validation_generator.batch_size,
                                          verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)

    errors = np.where(predicted_classes != ground_truth)[0]
    print("No of errors = {}/{}".format(len(errors), validation_generator.samples))




if __name__ == '__main__':
    main()