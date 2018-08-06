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
        seq_std = roi_sequence.std(axis=2)
        ret1, seq_std_norm_thresh = cv2.threshold(seq_std.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
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
    from utils import get_connected_components
    data_path = '/Users/shichao/workding_dir/data'
    img_vol = 120
    img_start_num = 1
    USE_CROP = False
    arbitrary_vol = 3
    seq_len = 3
    diff_step = 1
    diff_result = frame_diff_analysis(data_path=data_path,img_vol=img_vol,img_start_num=img_start_num,use_crop=USE_CROP,
                                      arbitrary_vol=arbitrary_vol,diff_step=diff_step)
    std_result = std_analysis(data_path=data_path,img_vol=img_vol,img_start_num=img_start_num,use_crop=USE_CROP,
                              arbitrary_vol=arbitrary_vol,seq_len=seq_len)


    print('use implementation')

    diff_labelled,_ = get_connected_components(diff_result)
    std_labelled,_ = get_connected_components(std_result)
    plt.subplot(121)
    plt.imshow(diff_labelled)
    plt.title('frame diffrence binary label')
    plt.subplot(122)
    plt.imshow(std_labelled)
    plt.title('std binary label')
    plt.show()







def main():
    result_analysis()

if __name__ == '__main__':
    main()