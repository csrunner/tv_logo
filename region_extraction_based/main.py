# -*- coding:utf-8 -*-
__author__ = 'shichao'

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from utils import readin_image,arbitrary_frame_diff,crop_image
from skimage.morphology import binary_dilation
from skimage.measure import regionprops
from skimage.measure import label


def frameDiff2std():
    readin_path = '/Users/shichao/workding_dir/data'
    img_num = 120
    img_stack = readin_image(readin_path=readin_path,maxim_num=img_num,start_num=1)
    CROP_IMAGE = False
    FRAME_DIFF_USE_MASK = True
    USE_NORM = False
    bbox_thresh = 700
    if CROP_IMAGE:
        img_stack = crop_image(img_stack)
    arbitrary_number = 5 # a small arbitrary_number gets better result
    seq_length = 3 # a small seq_length gets better result
    seq_multi_frame_diff = 255
    seq_multi_std = 255
    [dimx,dimy,_] = img_stack.shape
    is_v2 = cv2.__version__.startswith("2.")
    if is_v2:
        detector = cv2.SimpleBlobDetector()
    else:
        detector = cv2.SimpleBlobDetector_create()

    # take the frame difference approach
    diff_start = time.time()
    # diff_stack = frame_diff(flatten_img_stack,img_stack,use_flatten=False)
    diff_step = 1 # a small diff_step gets better result
    diff_stack = arbitrary_frame_diff(img_stack,step=arbitrary_number)
    if FRAME_DIFF_USE_MASK:
        print('frame_difference uses mask')
        diff_stack_length = diff_step
        diff_roi_sequence = np.zeros([dimx, dimy, diff_stack_length])
        for i in range(0,diff_stack.shape[2],diff_step):
            for j in range(diff_step):
                diff_roi_sequence[:, :, j] = diff_stack[:, :, i+j]
            diff_short_sum = np.sum(diff_roi_sequence, axis=2)
            if USE_NORM:
                diff_short_dst = np.zeros(diff_short_sum.shape)
                cv2.normalize(diff_short_sum, diff_short_dst, norm_type=cv2.NORM_MINMAX)
                diff_short_seq_std_norm_255 = (255 * diff_short_dst).astype(np.uint8)
            else:
                diff_short_dst = diff_short_sum
                diff_short_seq_std_norm_255 = diff_short_dst.astype(np.uint8)
            ret0, diff_short_seq_std_norm_thresh = cv2.threshold(diff_short_seq_std_norm_255, 0, 255,
                                                                       cv2.THRESH_OTSU)
            frame_diff_raw_result = diff_short_seq_std_norm_255
            diff_short_seq_std_norm_thresh_inv = 255 - diff_short_seq_std_norm_thresh
            seq_multi_frame_diff *= diff_short_seq_std_norm_thresh_inv / 255
            frame_diff_result = seq_multi_frame_diff

    else:
        difference_sum = np.sum(diff_stack, axis=2)
        frame_diff_dst = np.zeros(difference_sum.shape)
        if USE_NORM:
            cv2.normalize(difference_sum, frame_diff_dst, norm_type=cv2.NORM_MINMAX)
            frame_diff_seq_std_norm_255 = (255 * frame_diff_dst).astype(np.uint8)
        else:
            frame_diff_dst = difference_sum
            frame_diff_seq_std_norm_255 = frame_diff_dst.astype(np.uint8)
        ret0, frame_diff_seq_std_norm_thresh = cv2.threshold(frame_diff_seq_std_norm_255, 0, 255, cv2.THRESH_OTSU)

        # seq_std_norm_thresh = cv2.adaptiveThreshold(seq_std_norm_255, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                             cv2.THRESH_BINARY_INV, 11, 2)
        # seq_std_norm_thresh = cv2.adaptiveThreshold(seq_std_norm_255, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        #                                             cv2.THRESH_BINARY_INV, 11, 2)

        frame_diff_raw_result = frame_diff_seq_std_norm_255
        frame_diff_seq_std_norm_thresh_inv = 255 - frame_diff_seq_std_norm_thresh
        frame_diff_result = frame_diff_seq_std_norm_thresh_inv

    kernel = np.array([[1,0],[0,1]])
    frame_diff_result = binary_dilation(frame_diff_result)
    # key_points = detector.detect(frame_diff_result)
    # print(key_points)
    diff_end = time.time()
    diff_time = diff_end - diff_start





    # ------------------------------------------------
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


    std_start = time.time()

    stack_length = img_num / seq_length
    dimx,dimy,dimz = img_stack.shape
    if img_num>dimz:
        img_num = int(input("input a number less than {0}".format(dimz)))
    roi_sequence = np.zeros([dimx, dimy, seq_length])
    roi_std_stack = np.zeros([dimx, dimy, stack_length])

    std_stack_counter = 0

    for i in range(img_num - seq_length * arbitrary_number):
        for j in range(seq_length):
            roi_sequence[:, :, j] = img_stack[:, :, i + j * arbitrary_number]
        # if seq_counter == seq_length-1:
        seq_std = roi_sequence.std(axis=2)
        dst = np.zeros([dimx, dimy])
        cv2.normalize(seq_std, dst, cv2.NORM_MINMAX)
        seq_std_norm_255 = (255 * dst).astype(np.uint8)
        ret1, seq_std_norm_thresh = cv2.threshold(seq_std.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
        # print(ret)
        # seq_std_norm_thresh = cv2.adaptiveThreshold(seq_std_norm_255, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                             cv2.THRESH_BINARY_INV, 11, 2)
        # seq_std_norm_thresh = cv2.adaptiveThreshold(seq_std_norm_255, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        #                                             cv2.THRESH_BINARY_INV, 11, 2)
        seq_std_norm_thresh_inv = 255 - seq_std_norm_thresh
        # cv2.imshow('std',roi_sequence.std(axis=2))

        seq_counter = 0
        roi_std_stack[:, :, std_stack_counter] = seq_std_norm_thresh
        seq_multi_std *= seq_std_norm_thresh_inv / 255
        if std_stack_counter < stack_length - 1:
            std_stack_counter += 1
    std_end = time.time()
    std_time = std_end - std_start
    # ------------------------------------------------------------
    seq_multi_std = binary_dilation(seq_multi_std)
    # non_zero_frameDiff_ratio = len(np.nonzero(frame_diff_seq_std_norm_thresh_inv)[0]) / float(dimx * dimy)
    # non_zero_std_ratio = len(np.nonzero(seq_std_norm_thresh_inv)[0]) / float(dimx * dimy)
    # print('frame difference non zero ratio: {0} with elapsed time {1}'.format(non_zero_frameDiff_ratio, diff_time))
    # print('std non zero ratio: {0} with elapsed time {1}'.format(non_zero_std_ratio, std_time))

    print('frame difference non zero  elapsed time {0}'.format(diff_time))
    print('std non zero ratio elapsed time {0}'.format(std_time))
    # print(type(frame_diff_result))
    label_diff = label(frame_diff_result)
    label_std = label(seq_multi_std)
    props_diff = regionprops(label_diff)
    props_std = regionprops(label_std)
    # print(len(props_std))
    for prop_diff,prop_std in zip(props_diff,props_std):
        if prop_diff.bbox_area > bbox_thresh:
            print('diff centroid {0}'.format(prop_diff.centroid))
        if prop_std.bbox_area > bbox_thresh:
            print('std centroid {0}'.format(prop_std.centroid))
    plt.subplot(221)
    # plt.imshow(seq_std_norm_thresh_inv,cmap='jet')
    plt.imshow(frame_diff_result)
    plt.title('sum of {0} frames diff with step{1}'.format(img_num,arbitrary_number))

    # plt.subplot(323)
    # plt.imshow(diff_short_seq_std_norm_255)
    # plt.title('raw frame difference_otsu {0}'.format(arbitrary_number))

    plt.subplot(222)
    plt.hist(frame_diff_raw_result.flatten(), bins=256, normed=1, facecolor='green', alpha=0.75)
    # plt.imshow(frame_diff_seq_std_norm_255)
    plt.title('frame diff hist with thresh {0}'.format(ret0))



    plt.subplot(223)
    # plt.imshow(seq_std_norm_thresh_inv)
    plt.imshow(seq_multi_std)
    plt.title('std of {0} frames with step {1}'.format(img_num,arbitrary_number))

    # plt.subplot(324)
    # # plt.imshow(seq_std_norm_thresh_inv)
    # plt.imshow(seq_std_norm_255)
    # plt.title('raw standard devivation_otsu')

    plt.subplot(224)
    plt.hist(seq_std.flatten(), bins=256, normed=1, facecolor='green', alpha=0.75)
    plt.title('std hist with thresh {0}'.format(ret1))

    # plt.subplot(224)
    # plt.imshow(seq_std)
    # plt.title('standard devivation')
    plt.show()



def main():
    frameDiff2std()

if __name__ == '__main__':
    main()