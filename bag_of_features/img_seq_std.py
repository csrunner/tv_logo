# -*- coding:utf-8 -*-
__author__ = 'shichao'


import os
import numpy as np
import cv2
import glob



path_in = ''
path_out = ''
dirs = os.listdir(path_in)
SAVEMODE = False
DISPLAYMODE = True
for dir in dirs:
    img_paths = glob.glob(os.path.join(os.path.join(path_in,dir),'*.*'))
    anchor_img = cv2.imread(img_paths[0])
    [dim1,dim2,dim3] = anchor_img.shape
    img_gray = cv2.cvtColor(anchor_img,cv2.COLOR_RGB2GRAY)
    img_stack = np.zeros([dim1,dim2,10])
    dst = np.zeros([dim1,dim2])
    counter = 0
    stack_img_counter = 0
    for i in range(len(img_paths)):
        img = cv2.imread(img_paths[i])
        img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img_stack[:,:,counter] = img_gray
        if counter == 9:
            img_std = img_stack.var(axis=2)
            img_std_norm = cv2.normalize(img_std,dst,norm_type=cv2.NORM_MINMAX)
            ret, img_std_norm_thresh = cv2.threshold(255*img_std_norm,30,255,cv2.THRESH_BINARY_INV)
            kernel = np.ones([2,2])
            img_std_norm_thresh_erode = cv2.erode(img_std_norm_thresh,kernel=kernel)
            img_std_norm_thresh_erode_dilate = cv2.dilate(img_std_norm_thresh_erode,kernel=kernel)
            if stack_img_counter == 0:
                stack_img_std_norm_thresh = img_std_norm_thresh_erode_dilate
            elif stack_img_counter<9:
                stack_img_std_norm_thresh *= stack_img_std_norm_thresh
                stack_img_counter += 1
            else:
                stack_img_std_norm_thresh *= stack_img_std_norm_thresh
                stack_img_counter = 0
                if SAVEMODE:
                    path_out_folder = os.path.join(path_out,dir+'_erode_dilate_k2')
                    if not os.path.exists(path_out_folder):
                        os.mkdir(path_out_folder)
                    cv2.imwrite(os.path.join(path_out_folder,str(i).zfill(4)+'.jpg'),stack_img_std_norm_thresh)
                if DISPLAYMODE:
                    cv2.imshow('binarized img',stack_img_std_norm_thresh)
                    cv2.waitKey(3000)
            counter = 0