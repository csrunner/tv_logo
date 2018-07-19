# -*- coding:utf-8 -*-
__author__ = 'shichao'

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from utils import readin_image,arbitrary_frame_diff,crop_image

def frameDiff2std():
    readin_path = '/Users/shichao/workding_dir/data'
    img_stack = readin_image(readin_path=readin_path,start_num=1)
    CROP_IMAGE = False
    if CROP_IMAGE:
        img_stack = crop_image(img_stack)
    arbitrary_number = 20

    # take the frame difference approach
    diff_start = time.time()
    # diff_stack = frame_diff(flatten_img_stack,img_stack,use_flatten=False)
    diff_stack, flatten_diff_stack = arbitrary_frame_diff(img_stack,arbitrary_number=arbitrary_number)
    difference_sum = np.sum(diff_stack, axis=2)
    frame_diff_dst = np.zeros(difference_sum.shape)
    cv2.normalize(difference_sum, frame_diff_dst, norm_type=cv2.NORM_MINMAX)
    frame_diff_seq_std_norm_255 = (255 * frame_diff_dst).astype(np.uint8)
    ret, frame_diff_seq_std_norm_thresh = cv2.threshold(frame_diff_seq_std_norm_255, 0, 255, cv2.THRESH_OTSU)
    # seq_std_norm_thresh = cv2.adaptiveThreshold(seq_std_norm_255, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                             cv2.THRESH_BINARY_INV, 11, 2)
    # seq_std_norm_thresh = cv2.adaptiveThreshold(seq_std_norm_255, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                                             cv2.THRESH_BINARY_INV, 11, 2)
    frame_diff_seq_std_norm_thresh_inv = 255 - frame_diff_seq_std_norm_thresh
    diff_end = time.time()
    diff_time = diff_end - diff_start

    # take the std approach
    std_start = time.time()
    '''
    use standivation of all images
    seq_std = img_stack.std(axis=2)
    '''
    # seq_std = std(img_stack)
    # dst = np.zeros(img_stack.shape[:2])
    # cv2.normalize(seq_std, dst, cv2.NORM_MINMAX)
    # seq_std_norm_255 = (255 * dst).astype(np.uint8)
    # ret, seq_std_norm_thresh = cv2.threshold(seq_std_norm_255, 0, 255, cv2.THRESH_OTSU)
    # seq_std_norm_thresh_inv = 255 - seq_std_norm_thresh
    seq_length = 5
    img_num = 240
    stack_length = img_num / seq_length
    dimx,dimy,dimz = img_stack.shape
    if img_num>dimz:
        img_num = int(input("input a number less than {0}".format(dimz)))
    roi_sequence = np.zeros([dimx, dimy, seq_length])
    roi_std_stack = np.zeros([dimx, dimy, stack_length])
    seq_counter = 0
    std_stack_counter = 0
    seq_multi = 255
    for i in range(img_num - seq_length * arbitrary_number):
        for j in range(seq_length):
            roi_sequence[:, :, j] = img_stack[:, :, i + j * arbitrary_number]
        # if seq_counter == seq_length-1:
        seq_std = roi_sequence.std(axis=2)
        dst = np.zeros([dimx, dimy])
        cv2.normalize(seq_std, dst, cv2.NORM_MINMAX)
        seq_std_norm_255 = (255 * dst).astype(np.uint8)
        ret, seq_std_norm_thresh = cv2.threshold(seq_std_norm_255, 0, 255, cv2.THRESH_OTSU)
        # seq_std_norm_thresh = cv2.adaptiveThreshold(seq_std_norm_255, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                             cv2.THRESH_BINARY_INV, 11, 2)
        # seq_std_norm_thresh = cv2.adaptiveThreshold(seq_std_norm_255, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        #                                             cv2.THRESH_BINARY_INV, 11, 2)
        seq_std_norm_thresh_inv = 255 - seq_std_norm_thresh
        # cv2.imshow('std',roi_sequence.std(axis=2))

        seq_counter = 0
        roi_std_stack[:, :, std_stack_counter] = seq_std_norm_thresh
        seq_multi *= seq_std_norm_thresh_inv / 255
        if std_stack_counter < stack_length - 1:
            std_stack_counter += 1
    std_end = time.time()
    std_time = std_end - std_start

    non_zero_frameDiff_ratio = len(np.nonzero(frame_diff_seq_std_norm_thresh_inv)[0]) / float(dimx * dimy)
    non_zero_std_ratio = len(np.nonzero(seq_std_norm_thresh_inv)[0]) / float(dimx * dimy)
    print('frame difference non zero ratio: {0} with elapsed time {1}'.format(non_zero_frameDiff_ratio, diff_time))
    print('std non zero ratio: {0} with elapsed time {1}'.format(non_zero_std_ratio, std_time))

    plt.subplot(121)
    # plt.imshow(seq_std_norm_thresh_inv,cmap='jet')
    plt.imshow(frame_diff_seq_std_norm_thresh_inv)
    plt.title('sum of frame difference_otsu {0}'.format(arbitrary_number))

    # plt.subplot(222)
    # plt.imshow(difference_sum)
    # plt.title('sum of frame difference {0}'.format(arbitrary_number))

    plt.subplot(122)
    # plt.imshow(seq_std_norm_thresh_inv)
    plt.imshow(seq_multi)
    plt.title('standard devivation_otsu')

    # plt.subplot(224)
    # plt.imshow(seq_std)
    # plt.title('standard devivation')

def main():
    frameDiff2std()

if __name__ == '__main__':
    main()