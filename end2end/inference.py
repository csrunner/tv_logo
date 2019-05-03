#author__ = 'shichao'

from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras import models
from keras import layers
from keras import optimizers
import time

image_size = 224
val_batchsize = 1
validation_dir = '/home/shichao/data/TV_LOGO_TRAIN_VAL_TEST_tiny/val'
#mst4_layers_104classes.h5odel_path = 'vgg_finetune_last4.h5'
#model_path = 'vgg16_fintune_last4_layers_104classes.h5'
model_path = 'resnet50_last4_layers.h5'
model = load_model(model_path)

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# Get the filenames from the generator
fnames = validation_generator.filenames

# Get the ground truth from generator
ground_truth = validation_generator.classes

# Get the label to class mapping from the generator
label2index = validation_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())

# Get the predictions from the model using the generator
start = time.time()
predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
end = time.time()
print('elpased time: {0}'.format(end-start))
predicted_classes = np.argmax(predictions,axis=1)

errors = np.where(predicted_classes != ground_truth)[0]
'''
with open('./result.txt','a') as f:
    for i, j in zip(predicted_classes,ground_truth):
        f.write('predicted: {0} and gt: {1}\n'.format(i,j))
'''
with open('./result.txt','a') as f:
    #for i, j in zip(predicted_classes,ground_truth):
    f.write("accuracy = {}\n".format((1-float(len(errors))/validation_generator.samples)))
    f.write('experiment fps: {0} with val batch \n'.format(validation_generator.samples/(end-start),val_batchsize))
    f.write('experiment time: {0}\n'.format(time.localtime(time.time())))
    #f.write("No of errors = {}/{}\n".format(len(errors),validation_generator.samples))
    #f.write('experiment fps: {0} with val batch \n'.format(validation_generator.samples/(end-start),val_batchsize))
    #f.write('experiment fps: {0}\n'.format(time.time()))
#print("No of errors = {}/{}".format(len(errors),validation_generator.samples))

