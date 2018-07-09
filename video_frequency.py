# -*- coding:utf-8 -*-
__author__ = 'shichao'

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import cv2
import os
import tensorflow as tf

def readin_image(readin_path):
    img_paths = glob.glob(os.path.join(readin_path,'*.jpeg'))
    img_num = len(img_paths)
    [dimx,dimy,dimz] = cv2.imread(img_paths[0]).shape
    flatten_img_stack = np.zeros([dimx*dimy,img_num])
    img_stack = np.zeros([dimx,dimy,img_num])
    for i in range(img_num):
        img_bgr = cv2.imread(img_paths[i])
        img_stack[:,:,i] = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
        flatten_img_stack[:,i] = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY).flatten('C')
    return flatten_img_stack, img_stack, dimx, dimy,img_num


def low_pass(flatten_array,img_stack,dimx,dimy,use_flatten=True):
    b, a = signal.butter(3, 0.08, 'low')
    # b, a = signal.cheby1(3,Wn=0.08,btype='low')
    if use_flatten:
        low_pass_video_sig = np.zeros(flatten_array.shape)
        for i in range(dimx*dimy):

            sig_tmp = flatten_array[i,:]*np.random.random([flatten_array.shape[1],])
            sig = []
            for j in range(len(sig_tmp)):
                sig.append(float(sig_tmp[j]))
            # print(sig)
            sf = signal.filtfilt(b, a, sig)
            low_pass_video_sig[i,:] = sf
        return low_pass_video_sig
    else:
        low_pass_video_sig = signal.filtfilt(b,a,img_stack,axis=2)
        return low_pass_video_sig


def frame_diff(flatten_img_stack,img_stack,use_flatten=False):
    if use_flatten:
        [dimx,dimy] = flatten_img_stack.shape
        flatten_diff_stack = np.zeros([dimx,dimy-1])
        previous = flatten_img_stack[:,0]
        for i in range(1,dimy):
            this = flatten_img_stack[:,i]
            flatten_img_stack[:,i-1] = this-previous
        return flatten_diff_stack
    else:
        [dimx,dimy,dimz] = img_stack.shape
        diff_stack = np.zeros([dimx,dimy,dimz-1])
        previous = img_stack[:,:,0]
        for i in range(1,dimz):
            this = img_stack[:,:,i]
            tmp = this-previous
            diff_stack[:,:,i - 1] = abs(tmp)
            previous = this
        return diff_stack


def img_freq_analysis():
    PLT_SHOW = True
    readin_path = '/Users/shichao/Downloads'
    [flatten_img_stack, img_stack, dimx, dimy, img_num] = readin_image(readin_path)
    diff_stack = frame_diff(flatten_img_stack,img_stack,use_flatten=False)
    difference_sum = np.sum(diff_stack,axis=2)
    frame_diff_dst = np.zeros(difference_sum.shape)
    cv2.normalize(difference_sum, frame_diff_dst, norm_type=cv2.NORM_MINMAX)
    frame_diff_seq_std_norm_255 = (255 * frame_diff_dst).astype(np.uint8)
    ret, frame_diff_seq_std_norm_thresh = cv2.threshold(frame_diff_seq_std_norm_255, 0, 255, cv2.THRESH_OTSU)
    # seq_std_norm_thresh = cv2.adaptiveThreshold(seq_std_norm_255, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                             cv2.THRESH_BINARY_INV, 11, 2)
    # seq_std_norm_thresh = cv2.adaptiveThreshold(seq_std_norm_255, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                                             cv2.THRESH_BINARY_INV, 11, 2)
    frame_diff_seq_std_norm_thresh_inv = 255 - frame_diff_seq_std_norm_thresh

    # use standivation
    seq_std = diff_stack.std(axis=2)
    dst = np.zeros(difference_sum.shape)
    cv2.normalize(seq_std, dst, cv2.NORM_MINMAX)
    seq_std_norm_255 = (255 * dst).astype(np.uint8)
    ret, seq_std_norm_thresh = cv2.threshold(seq_std_norm_255, 0, 255, cv2.THRESH_OTSU)
    seq_std_norm_thresh_inv = 255 - seq_std_norm_thresh

    non_zero_frameDiff_ratio = len(np.nonzero(frame_diff_seq_std_norm_thresh_inv)[0])/float(dimx*dimy)
    non_zero_std_ratio = len(np.nonzero(seq_std_norm_thresh_inv)[0])/float(dimx*dimy)
    print('frame difference non zero ratio: {0}'.format(non_zero_frameDiff_ratio))
    print('std non zero ratio: {0}'.format(non_zero_std_ratio))



    if PLT_SHOW:
        plt.subplot(221)
        # plt.imshow(seq_std_norm_thresh_inv,cmap='jet')
        plt.imshow(frame_diff_seq_std_norm_thresh_inv)
        plt.title('frame difference_otsu')

        plt.subplot(222)
        plt.imshow(difference_sum)
        plt.title('frame difference')



        plt.subplot(223)
        plt.imshow(seq_std_norm_thresh_inv)
        plt.title('standard devivation_otsu')

        plt.subplot(224)
        plt.imshow(seq_std)
        plt.title('standard devivation')


        plt.show()
    else:
        cv2.imshow('difference',seq_std_norm_thresh_inv)
        cv2.waitKey(5000)
        cv2.destroyWindow('difference')

def flatten_img_freq_analysis():
    readin_path = '/Users/shichao/Downloads'
    [flatten_img_stack,img_stack,dimx,dimy,img_num] = readin_image(readin_path)
    flatten_diff_stack = frame_diff(flatten_img_stack)
    low_pass_video_sig = low_pass(flatten_img_stack,img_stack,dimx,dimy,True)
    low_pass_img_stack = low_pass_video_sig.reshape(dimx,dimy,img_num)
    axis_x = np.linspace(0, low_pass_video_sig.shape[1], num=low_pass_video_sig.shape[1])
    # 频率为5Hz的正弦信号
    plt.subplot(221)
    # 15860 21670 31253 22123
    logo_point = 22123
    non_logo_point = 4500
    plt.plot(axis_x, flatten_img_stack[logo_point,:])
    plt.title(u'logo original')
    plt.axis('tight')

    plt.subplot(222)
    plt.plot(axis_x, low_pass_video_sig[logo_point,:])
    plt.title(u'logo low pass')
    plt.axis('tight')

    plt.subplot(223)
    plt.plot(axis_x, flatten_img_stack[non_logo_point, :])
    plt.title(u'non-logo original')
    plt.axis('tight')

    plt.subplot(224)
    plt.plot(axis_x, low_pass_video_sig[non_logo_point, :])
    plt.title(u'non-logo low pass')
    plt.axis('tight')
    plt.show()
    # for i in range(low_pass_img_stack.shape[2]):
    #     cv2.imshow('low_pass{0}'.format(i),low_pass_img_stack[:,:,i])
    #     cv2.waitKey(5000)
    #     cv2.destroyWindow('low_pass{0}'.format(i))

def main():
    img_freq_analysis()

if __name__ == '__main__':
    main()