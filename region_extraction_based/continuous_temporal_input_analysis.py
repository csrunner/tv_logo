# -*- coding:utf-8 -*-
__author__ = 'shichao'

import cv2
import glob
import os
import time
import numpy as np
# from utils import plot_img_hist

# path_in = '/home/shichao/Documents/Image-classification/raw_data'
path_out = '/home/shichao/data/temporal_std_thresh'
path_in = '/Users/shichao/Downloads/transport'
dirs = os.listdir(path_in)
SAVETOPATH = False
SHOW = True
for dir in dirs:
    img_paths = glob.glob(os.path.join(os.path.join(path_in,dir),'*.*'))
    img = cv2.imread(img_paths[0])
    [dim1,dim2,dim3] = img.shape
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    stack = 100
    stack2 = 5
    img_stack = np.zeros([dim1,dim2,stack])
    dst = np.zeros([dim1,dim2])
    counter = 0
    stack_img_counter = 0
    for i in range(len(img_paths)):
        img = cv2.imread(os.path.join(os.path.join(path_in,dir),str(i+1).zfill(4)+'.*'))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ret_tmp, img_gray_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
        # print(type(img_gray))
        # print(img_gray.ndim)
        # plot_img_hist(img_gray)
        # cv2.imshow('the {0}th image'.format(i),img_gray)
        # cv2.waitKey(5000)
        img_stack[:, :, counter] = img_gray
        if counter == stack-1:
            counter = 0
            img_std = img_stack.std(axis=2)
            cv2.imshow('std',img_std)
            cv2.waitKey(3000)
            img_std_norm = cv2.normalize(img_std, dst, norm_type=cv2.NORM_MINMAX)
            # plot_img_hist(img_std_norm)
            img_std_norm_255 = 255*np.ones(img_std_norm.shape)*img_std_norm
            img_std_norm_255 = img_std_norm_255.astype(np.uint8)
            # cv2.imwrite('/home/shichao/thresholdtest.jpg',img_std_norm_255)
            # print(type(img_std_norm_255))
            # plot_img_hist(img_std_norm_255)
            # print(np.max(np.max(img_std_norm)))
            # cv2.imshow('variance',img_std_norm_255)
            # cv2.waitKey(1000)
            # ret, img_std_norm_thresh = cv2.threshold(255 * img_std_norm, 30, 255, cv2.THRESH_BINARY_INV)
            ret, img_std_norm_thresh = cv2.threshold(img_std_norm_255, 0, 255, cv2.THRESH_OTSU)

            # cv2.imshow('img_std',img_std_norm_thresh)
            # cv2.waitKey(1000)
            kernel = np.ones([2,2])
            # kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])
            img_std_norm_thresh_erode = cv2.erode(img_std_norm_thresh,kernel=kernel)
            img_std_norm_thresh_erode_dilate = cv2.dilate(img_std_norm_thresh_erode,kernel=kernel)
            if stack_img_counter == 0:
                stack_img_std_norm_thresh = img_std_norm_thresh_erode_dilate
                stack_img_counter += 1
            elif stack_img_counter < stack2-1:
                stack_img_std_norm_thresh *= img_std_norm_thresh_erode_dilate
                stack_img_counter += 1
            else:
                stack_img_std_norm_thresh *= img_std_norm_thresh_erode_dilate
                print('the {0}th iteration of {1}'.format(stack_img_counter,'stack_img_counter'))
                stack_img_counter = 0
                if SAVETOPATH:
                    path_out_folder = os.path.join(path_out, dir + '_erode_dilate_2')
                    if not os.path.exists(path_out_folder):
                        os.mkdir(path_out_folder)
                    cv2.imwrite(os.path.join(path_out_folder, str(i).zfill(4) + '.jpg'), stack_img_std_norm_thresh)
                if SHOW:
                    cv2.imshow('bin img',stack_img_std_norm_thresh)
                    cv2.waitKey(3000)

        counter += 1
        # counter += 120
