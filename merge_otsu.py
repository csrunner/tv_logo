# -*- coding:utf-8 -*-
__author__ = 'shichao'


import os
import glob
import cv2
import numpy as np


PIXEL_TEST = False
PATCH_TEST = False
IMAGE_TEST = True

if PIXEL_TEST:
    # img_path = '/Users/shichao/Downloads/transport/dataset/0060.jpeg'
    # img = cv2.imread(img_path)
    # img_part = img[27:68,30:58,:]

    path_in = '/Users/shichao/Downloads/transport'
    # path_out = '/home/shichao/data/temporal_std_thresh'

    dirs = os.listdir(path_in)
    SAVETOPATH = False
    SHOW = True

    seq_counter = 0
    stack_counter = 0


    for dir in dirs:
        file_amount = len(glob.glob(os.path.join(os.path.join(path_in,dir),'*.*')))
        seq_length = 5
        stack_length = file_amount/seq_length

        roi_sequence = np.zeros([seq_length, 1])
        roi_stack = np.zeros([stack_length, 1])

        for i in range(file_amount):
            img_path = glob.glob(os.path.join(os.path.join(path_in,dir),str(i+1).zfill(4)+'.*'))
            img = cv2.imread(img_path[0])
            dimy,dimx,dimz = img.shape
            # y = np.random.randint(68,dimy)
            # x = np.random.randint(58,dimx)
            y = 35
            x = 30
            # roi_sequence[seq_counter] = img[30,35,2]

            # roi_sequence[seq_counter] = img[38, 50, 2] # logo cruppted by youku
            # roi_sequence[seq_counter] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[38, 50]

            # roi_sequence[seq_counter] = img[58, 35, 2] # low std in the logo region

            # roi_sequence[seq_counter] = img[y, x, 2]
            roi_sequence[seq_counter] = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)[y,x]

            if seq_counter == seq_length-1:
                roi_stack[stack_counter] = roi_sequence.std()
                print('the frame sequence std is {0}'.format(roi_sequence.std()))
                seq_counter = 0
                if stack_counter < stack_length-1:
                    stack_counter += 1
            else:
                seq_counter += 1
        print('the std of seq_std is {0}'.format(roi_stack.std()))
        print('the median of seq_std is {0}'.format(np.median(roi_stack)))

            # img_part = img[27:68,30:58,:]
            # print(img[30,35,:]) # cruppted by the youku logo from approximately 23 frame to 40 frame
            # cv2.imshow('cropped',img_part)
            # cv2.waitKey(500)
            # cv2.destroyWindow('cropped')
if PATCH_TEST:
    pass
if IMAGE_TEST:
    path_in = '/Users/shichao/Downloads/transport'
    # path_out = '/home/shichao/data/temporal_std_thresh'

    dirs = os.listdir(path_in)
    SAVETOPATH = False
    SHOW = True

    seq_counter = 0
    stack_counter = 0

    seq_counter = 0
    std_stack_counter = 0
    seq_multi = 255
    for dir in dirs:
        if dir.startswith('.'):
            continue
        file_amount = len(glob.glob(os.path.join(os.path.join(path_in, dir), '*.*')))
        seq_length = 5
        stack_length = file_amount / seq_length
        dimy,dimx,dimz = cv2.imread(glob.glob(os.path.join(os.path.join(path_in, dir), '*.*'))[0]).shape
        roi_sequence = np.zeros([dimy,dimx,seq_length])
        roi_std_stack = np.zeros([dimy,dimx,stack_length])

        for i in range(file_amount):
            img = cv2.imread(glob.glob(os.path.join(os.path.join(path_in, dir), str(i + 1).zfill(4) + '.*'))[0])

            roi_sequence[:,:,seq_counter] = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            if seq_counter == seq_length-1:
                seq_std = roi_sequence.std(axis=2)
                roi_std_stack[:,:,std_stack_counter] = seq_std
                seq_counter = 0
                if std_stack_counter < stack_length-1:
                    std_stack_counter += 1

                # seq_std_norm_thresh = cv2.adaptiveThreshold(seq_std_norm_255, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                #                                             cv2.THRESH_BINARY_INV, 11, 2)
                # seq_std_norm_thresh = cv2.adaptiveThreshold(seq_std_norm_255, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                #                                             cv2.THRESH_BINARY_INV, 11, 2)
                # cv2.imshow('std',roi_sequence.std(axis=2))
                # cv2.imshow('std',seq_std_norm_thresh_inv)
                # cv2.waitKey(1500)
                # cv2.destroyWindow('std')

            else:
                seq_counter += 1
        roi_std_stack_median = np.median(roi_std_stack,axis=2)
        dst = np.zeros([dimy, dimx])
        cv2.normalize(roi_std_stack_median, dst, cv2.NORM_MINMAX)
        seq_std_median_norm_255 = (255 * dst).astype(np.uint8)
        ret, seq_std_norm_thresh = cv2.threshold(seq_std_median_norm_255, 0, 255, cv2.THRESH_OTSU)
        cv2.imshow('before otsu',roi_std_stack_median)
        cv2.waitKey(20000)
        cv2.destroyWindow('before otsu')
        cv2.imshow('logo extraction',seq_std_norm_thresh)
        # cv2.imshow('logo extraction',seq_multi)
        cv2.waitKey(20000)
        cv2.destroyWindow('logo extraction')