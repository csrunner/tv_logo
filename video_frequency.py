# -*- coding:utf-8 -*-
__author__ = 'shichao'

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import cv2
import os

def readin_image(readin_path):
    img_paths = glob.glob(os.path.join(readin_path,'*.jpg'))
    img_num = len(img_paths)
    [dimx,dimy,dimz] = cv2.imread(img_paths[0]).shape
    img_stack = np.zeros([dimx*dimy,img_num])
    for i in range(img_num):
        img_bgr = cv2.imread(img_paths[i])
        img_stack[:,i] = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY).flatten('F')
    return img_stack, dimx, dimy,img_num


def low_pass(flatten_array,dimx,dimy):
    b, a = signal.butter(3, 0.08, 'low')
    low_pass_video_sig = np.zeros(flatten_array.shape)
    for i in range(dimx*dimy):
        sig_tmp = flatten_array[i,:]
        print(type(flatten_array))
        print(len(sig_tmp))
        sig = []
        for j in range(len(sig_tmp)):
            sig.append(sig_tmp[j])
        print(type(sig))
        # print(sig)
        sf = signal.filtfilt(b, a, sig)
        low_pass_video_sig[i,:] = sf
    return low_pass_video_sig



def main():
    readin_path = '/Users/shichao/Downloads'
    [img_stack,dimx,dimy,img_num] = readin_image(readin_path)
    low_pass_video_sig = low_pass(img_stack,dimx,dimy)
    low_pass_img_stack = low_pass_video_sig.reshape(dimx,dimy,img_num)
    cv2.imshow('low_pass',low_pass_img_stack[:,:,0])
    cv2.waitKey(5000)


if __name__ == '__main__':
    main()