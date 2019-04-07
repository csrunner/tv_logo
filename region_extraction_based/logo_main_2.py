# -*- coding:utf-8 -*-
__author__ = 'shichao'


import cv2
import numpy as np
import glob
import os


def logo_detection(img):
    pred = 0
    return pred

def find_label(path):
    label = 0
    return label

def one_img_detection(img_path):


    row = 3
    col = 3

    img = cv2.imread(img_path)
    [dimx, dimy, dimc] = img.shape
    # label = find_label(img_path)
    left_top  = img[:dimx/row,:dimy/col,:]
    right_top = img[:dimx/row,:-dimy/col,:]
    left_bottom = img[:-dimx/row,:dimy/col,:]
    right_bottom = img[:-dimx/row,:-dimy/col,:]
    pred_lt = logo_detection(left_top)
    if pred_lt == 0:
        pred_rt = logo_detection(right_top)
        if pred_rt == 0:
            pred_lb = logo_detection(left_bottom)
            if pred_lb == 0:
                pred_rb = logo_detection(right_bottom)
                pred = 0 if pred_rb == 0 else pred_rb
            else:
                pred = pred_lb
        else:
            pred = pred_rt
    else:
        pred = pred_lt
    return pred
    # return True if pred == label else False

def one_folder_detection(folder_path):
    start_num = 1
    img_paths = glob.glob(os.path.join(folder_path, '*.*'))
    img_vol = len(img_paths)
    filename = os.path.basename(img_paths[0])
    name, ext = os.path.splitext(filename)
    zfill_num = len(name)
    label = find_label(folder_path)
    correct = 0
    wrong = 0
    for i in range(img_vol):
        img_path = os.path.join(folder_path,str(i+start_num).zfill(zfill_num)+ext)
        pred = one_img_detection(img_path)
        if pred == label:
            correct += 1
        else:
            wrong += 1

    accuracy = correct/float(correct+wrong)
    return accuracy


def all_folder_detection(root_path):
    acc = 0
    folder_num = 0
    for curdir in os.listdir(root_path):

        if os.path.isdir(curdir):
            folder_num += 1
            folder_path = os.path.join(root_path,curdir)
            acc_one_folder = one_folder_detection(folder_path)
            acc += acc_one_folder
    avr_acc = acc/folder_num











def main():
    data_path = '/Users/shichao/workding_dir/data'
    img_vol = 120
    img_start_num = 1
    USE_CROP = True
    one_img_detection(data_path)
