# -*- coding:utf-8 -*-
__author__ = 'shichao'

'''
SOURCE:
--root
     |---station1
            |-----00001.jpg
     |---station2
     


DESTINATION:
--root
     |---train
             |---station1
                    |-----00001.jpg
             |---station2
     |---validation
             |---station1
                    |-----00001.jpg
             |---station2
     |---test
             |---station1
                    |-----00001.jpg
             |---station2
             
'''

import os
import shutil
import sys
import glob

source_root_path = '../data/TV_LOGO_MERGE'
dest_root_path = '../data/TV_LOGO_TRAIN_VAL_TEST'
dest_train_path = os.path.join(dest_root_path,'train')
dest_val_path = os.path.join(dest_root_path,'val')
dest_test_path = os.path.join(dest_root_path,'test')

ext_list = ['.jpg','.jpeg','.png']
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

row = 3
col = 3

station_lists = os.listdir(source_root_path)
for station in station_lists:
    station_path = os.path.join(source_root_path,station)

    if os.path.isdir(station_path):

        files_list = os.listdir(station_path)
        cur_ext = os.path.splitext(files_list[0])[1]
        file_quantity = len(files_list)
        for i in range(file_quantity):
            file_path = os.path.join(station_path,files_list[i])
            if i < file_quantity*train_ratio:
                try:
                    img = cv2.imread(file_path)
                    [dimx,dimy,dimc] = img.shape
                    cropped_img = np.zeros([2*dimx/row,2*dimy/col,dimc])
                    cropped_img[:dimx/row,:dimy/col,:,i]  = img_seq[:dimx/row,:dimy/col,:,i]
                    cropped_img[:dimx/row,:-dimy/col,:,i] = img_seq[:dimx/row,:-dimy/col,:,i]
                    cropped_img[:-dimx/row,:dimy/col,:,i] = img_seq[:-dimx/row,:dimy/col,:,i]
                    cropped_img[:-dimx/row,:-dimy/col,:,i] = img_seq[:-dimx/row,:-dimy/col,:,i]
                    resized_cropped_img = cv2.resize(cropped_img,(640,480))
                    cv2.imwrite(os.path.join(os.path.join(os.path.join(dest_root_path,'train'),station),files_list[i]),resized_cropped_img)
                    #shutil.copyfile(file_path,os.path.join(os.path.join(os.path.join(dest_root_path,'train'),station),files_list[i]))
                except IOError:
                    os.mkdir(os.path.join(os.path.join(dest_root_path,'train'),station))
                    img = cv2.imread(file_path)
                    [dimx,dimy,dimc] = img.shape
                    cropped_img = np.zeros([2*dimx/row,2*dimy/col,dimc])
                    cropped_img[:dimx/row,:dimy/col,:,i]  = img_seq[:dimx/row,:dimy/col,:,i]
                    cropped_img[:dimx/row,:-dimy/col,:,i] = img_seq[:dimx/row,:-dimy/col,:,i]
                    cropped_img[:-dimx/row,:dimy/col,:,i] = img_seq[:-dimx/row,:dimy/col,:,i]
                    cropped_img[:-dimx/row,:-dimy/col,:,i] = img_seq[:-dimx/row,:-dimy/col,:,i]
                    resized_cropped_img = cv2.resize(cropped_img,(640,480))
                    cv2.imwrite(os.path.join(os.path.join(os.path.join(dest_root_path,'train'),station),files_list[i]),resized_cropped_img)
                    #shutil.copyfile(file_path,
                    #                os.path.join(os.path.join(os.path.join(dest_root_path, 'train'), station),
                    #                             files_list[i]))

            elif i >= file_quantity*train_ratio and i < file_quantity*(train_ratio+val_ratio):
                try:
                    img = cv2.imread(file_path)
                    [dimx,dimy,dimc] = img.shape
                    cropped_img = np.zeros([2*dimx/row,2*dimy/col,dimc])
                    cropped_img[:dimx/row,:dimy/col,:,i]  = img_seq[:dimx/row,:dimy/col,:,i]
                    cropped_img[:dimx/row,:-dimy/col,:,i] = img_seq[:dimx/row,:-dimy/col,:,i]
                    cropped_img[:-dimx/row,:dimy/col,:,i] = img_seq[:-dimx/row,:dimy/col,:,i]
                    cropped_img[:-dimx/row,:-dimy/col,:,i] = img_seq[:-dimx/row,:-dimy/col,:,i]
                    resized_cropped_img = cv2.resize(cropped_img,(640,480))
                    cv2.imwrite(os.path.join(os.path.join(os.path.join(dest_root_path,'train'),station),files_list[i]),resized_cropped_img)
                    #shutil.copyfile(file_path,os.path.join(os.path.join(os.path.join(dest_root_path,'val'),station),files_list[i]))
                except IOError:
                    os.mkdir(os.path.join(os.path.join(dest_root_path,'val'),station))
                    img = cv2.imread(file_path)
                    [dimx,dimy,dimc] = img.shape
                    cropped_img = np.zeros([2*dimx/row,2*dimy/col,dimc])
                    cropped_img[:dimx/row,:dimy/col,:,i]  = img_seq[:dimx/row,:dimy/col,:,i]
                    cropped_img[:dimx/row,:-dimy/col,:,i] = img_seq[:dimx/row,:-dimy/col,:,i]
                    cropped_img[:-dimx/row,:dimy/col,:,i] = img_seq[:-dimx/row,:dimy/col,:,i]
                    cropped_img[:-dimx/row,:-dimy/col,:,i] = img_seq[:-dimx/row,:-dimy/col,:,i]
                    resized_cropped_img = cv2.resize(cropped_img,(640,480))
                    cv2.imwrite(os.path.join(os.path.join(os.path.join(dest_root_path,'train'),station),files_list[i]),resized_cropped_img)
                    #shutil.copyfile(file_path,
                    #                os.path.join(os.path.join(os.path.join(dest_root_path, 'val'), station),
                    #                             files_list[i]))

            else:
                try:
                    img = cv2.imread(file_path)
                    [dimx,dimy,dimc] = img.shape
                    cropped_img = np.zeros([2*dimx/row,2*dimy/col,dimc])
                    cropped_img[:dimx/row,:dimy/col,:,i]  = img_seq[:dimx/row,:dimy/col,:,i]
                    cropped_img[:dimx/row,:-dimy/col,:,i] = img_seq[:dimx/row,:-dimy/col,:,i]
                    cropped_img[:-dimx/row,:dimy/col,:,i] = img_seq[:-dimx/row,:dimy/col,:,i]
                    cropped_img[:-dimx/row,:-dimy/col,:,i] = img_seq[:-dimx/row,:-dimy/col,:,i]
                    resized_cropped_img = cv2.resize(cropped_img,(640,480))
                    cv2.imwrite(os.path.join(os.path.join(os.path.join(dest_root_path,'train'),station),files_list[i]),resized_cropped_img)
                    #shutil.copyfile(file_path,os.path.join(os.path.join(os.path.join(dest_root_path,'test'),station),files_list[i]))
                except IOError:
                    os.mkdir(os.path.join(os.path.join(dest_root_path,'test'),station))
                    img = cv2.imread(file_path)
                    [dimx,dimy,dimc] = img.shape
                    cropped_img = np.zeros([2*dimx/row,2*dimy/col,dimc])
                    cropped_img[:dimx/row,:dimy/col,:,i]  = img_seq[:dimx/row,:dimy/col,:,i]
                    cropped_img[:dimx/row,:-dimy/col,:,i] = img_seq[:dimx/row,:-dimy/col,:,i]
                    cropped_img[:-dimx/row,:dimy/col,:,i] = img_seq[:-dimx/row,:dimy/col,:,i]
                    cropped_img[:-dimx/row,:-dimy/col,:,i] = img_seq[:-dimx/row,:-dimy/col,:,i]
                    resized_cropped_img = cv2.resize(cropped_img,(640,480))
                    cv2.imwrite(os.path.join(os.path.join(os.path.join(dest_root_path,'train'),station),files_list[i]),resized_cropped_img)
                    #shutil.copyfile(file_path,
                    #                os.path.join(os.path.join(os.path.join(dest_root_path, 'test'), station),
                    #                             files_list[i]))

        # for files in os.listdir(station_path):
        #     file_path = os.path.join(station_path,files)







