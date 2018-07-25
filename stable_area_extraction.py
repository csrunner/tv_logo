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


def frame_diff_analysis(data_path,img_vol,img_start_num,use_crop,arbitrary_vol,diff_step):
    '''
    input the data dir and get the stable roi region using frame difference
    :param data_path:
    :param img_vol:
    :param img_start_num:
    :param use_crop:
    :param bbox_thresh:
    :param arbitrary_vol:
    :param seq_len:
    :param diff_step:
    :return:
    '''
    img_stack = readin_image(readin_path=data_path, maxim_num=img_vol, start_num=img_start_num)
    if use_crop:
        img_stack = crop_image(img_stack)
    seq_multi_frame_diff = 255
    [dimx, dimy, _] = img_stack.shape
    start = time.time()
    diff_stack = arbitrary_frame_diff(img_stack, step=arbitrary_vol)
    diff_stack_length = diff_step
    diff_roi_sequence = np.zeros([dimx, dimy, diff_stack_length])
    for i in range(0, diff_stack.shape[2], diff_step):
        for j in range(diff_step):
            diff_roi_sequence[:, :, j] = diff_stack[:, :, i + j]
        diff_short_sum = np.sum(diff_roi_sequence, axis=2)
        diff_short_seq_diff_255 = diff_short_sum.astype(np.uint8)
        ret0, diff_short_seq_diff_thresh = cv2.threshold(diff_short_seq_diff_255, 0, 255,
                                                             cv2.THRESH_OTSU)
        diff_short_seq_std_norm_thresh_inv = 255 - diff_short_seq_diff_thresh
        seq_multi_frame_diff *= diff_short_seq_std_norm_thresh_inv / 255
        frame_diff_result = seq_multi_frame_diff
    frame_diff_result = binary_dilation(frame_diff_result)
    end = time.time()
    elapsed_time = end-start
    print('frame difference elapsed time: {0}'.format(elapsed_time))
    return frame_diff_result


def std_analysis(data_path,img_vol,img_start_num,use_crop,arbitrary_vol,seq_len):
    '''
    input the data dir and get the stable roi region using standard deviation
    :param data_path:
    :param img_vol:
    :param img_start_num:
    :param use_crop:
    :param bbox_thresh:
    :param arbitrary_vol:
    :param seq_len:
    :return:
    '''
    img_stack = readin_image(readin_path=data_path, maxim_num=img_vol, start_num=img_start_num)
    if use_crop:
        img_stack = crop_image(img_stack)
    seq_multi_std = 255
    start = time.time()
    stack_length = img_vol / seq_len
    dimx, dimy, dimz = img_stack.shape
    if img_vol > dimz:
        img_vol = int(input("input a number less than {0}".format(dimz)))
    roi_sequence = np.zeros([dimx, dimy, seq_len])
    roi_std_stack = np.zeros([dimx, dimy, stack_length])

    std_stack_counter = 0

    for i in range(img_vol - seq_len * arbitrary_vol):
        for j in range(seq_len):
            roi_sequence[:, :, j] = img_stack[:, :, i + j * arbitrary_vol]
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
        roi_std_stack[:, :, std_stack_counter] = seq_std_norm_thresh
        seq_multi_std *= seq_std_norm_thresh_inv / 255
        if std_stack_counter < stack_length - 1:
            std_stack_counter += 1
    end = time.time()
    elapsed_time = end-start
    print('std elapsed time: {0}'.format(elapsed_time))
    seq_multi_std = binary_dilation(seq_multi_std)
    return seq_multi_std


def result_analysis():
    data_path = '/Users/shichao/workding_dir/data'
    img_vol = 120
    img_start_num = 1
    USE_CROP = False
    arbitrary_vol = 5
    seq_len = 3
    diff_step = 1
    bbox_thresh = 700
    diff_result = frame_diff_analysis(data_path=data_path,img_vol=img_vol,img_start_num=img_start_num,use_crop=USE_CROP,
                                      arbitrary_vol=arbitrary_vol,diff_step=diff_step)
    std_result = std_analysis(data_path=data_path,img_vol=img_vol,img_start_num=img_start_num,use_crop=USE_CROP,
                              arbitrary_vol=arbitrary_vol,seq_len=seq_len)
    label_diff = label(diff_result)
    label_std = label(std_result)
    props_diff = regionprops(label_diff)
    props_std = regionprops(label_std)

    for prop_diff, prop_std in zip(props_diff, props_std):
        if prop_diff.bbox_area > bbox_thresh:
            print('diff centroid {0}'.format(prop_diff.centroid))
        if prop_std.bbox_area > bbox_thresh:
            print('std centroid {0}'.format(prop_std.centroid))

    plt.subplot(121)
    plt.imshow(diff_result)
    plt.title('sum of {0} frames diff with step{1}'.format(img_vol, arbitrary_vol))



    plt.subplot(122)
    plt.imshow(std_result)
    plt.title('std of {0} frames with step {1}'.format(img_vol, arbitrary_vol))

    plt.show()


def main():
    result_analysis()

if __name__ == '__main__':
    main()
