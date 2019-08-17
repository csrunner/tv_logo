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
    elif args.net == 'shufflenet_v2':
        from shufflenet_v2 import ShuffleNetV2
        net = ShuffleNetV2
    elif args.net == 'mobilenet_v2':
        from keras.applications.mobilenet_v2 import MobileNetV2
        net = MobileNetV2
    else:
        raise RuntimeError('the network is not supported yet')
    return net

def main():




    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    # parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    # parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    # parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    # parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-path', type=str, default='./', help='root data path')
    parser.add_argument('-e',type=int,default=20,help='epoch')
    parser.add_argument('-width',type=int,default=224,help='image width')
    parser.add_argument('-height',type=int,default=224,help='image height')
    args = parser.parse_args()

    root = args.path
    train_dir = os.path.join(root,'train')
    validation_dir = os.path.join(root,'val')
    num_classes = len(os.listdir(train_dir))
    epoch = args.e
    image_size1 = args.width
    image_size2 = args.height

    net = get_network(args)
    if args.net in ['mobilenet_v2']:
        conv = net(input_shape=(args.width, args.height, 3), classes=num_classes, weights=None)
    else:
        conv = net(input_shape=(args.width, args.height, 3), classes=num_classes)
    model = models.Sequential()
    model.add(conv)

    if args.net in ('squeezenet'):
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
    num_training_images = len(train_generator.filenames)
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
    import time
    now = int(time.time())
    timeStruct = time.localtime(now)
    strTime = time.strftime("%Y-%m-%d-%H-%M", timeStruct)
    model.save('{0}_{1}_classes_{2}_w{3}h{4}.h5'.format(args.net,num_classes,timeStruct,args.w,args.h))

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