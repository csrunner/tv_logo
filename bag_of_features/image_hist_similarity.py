# -*- coding: utf-8 -*-
# __author__ = 'shichao'

import cv2
import numpy as np
import os
import os.path
# from matplotlib import pyplot as plt
import glob
import time

def get_histGBR(path):
    img = cv2.imread(path)

    pixal = img.shape[0] * img.shape[1]
    total = np.array([0])
    for i in range(3):
        histSingle = cv2.calcHist([img], [i], None, [256], [0, 256])
        total = np.vstack((total, histSingle))
    return (total, pixal)

def hist_similar(lhist, rhist, lpixal,rpixal):
    rscale = rpixal/lpixal
    rhist = rhist/rscale
    assert len(lhist) == len(rhist)
    likely =  sum(1 - (0 if l == r else float(abs(l - r)) / max(l, r)) for l, r in zip(lhist, rhist)) / len(lhist)
    if likely == 1.0:
        return [1.0]
    return likely



def hist_similar(lhist, rhist, lpixal,rpixal):
    eps = 1e-6
    rscale = rpixal/(lpixal+eps)
    rhist = rhist/(rscale+eps)
    assert len(lhist) == len(rhist)
    likely = sum(1 - (0 if l == r else float(abs(l - r)) / (max(l, r))+eps) for l, r in zip(lhist, rhist)) / len(lhist)
    if likely == 1.0:
        return [1.0]
    return likely


def get_histGBR(path):
    img_tmp = cv2.imread(path)
    img = img_tmp
    pixal = img.shape[0] * img.shape[1]
    # print(pixal)
    # scale = pixal/100000.0
    # print(scale)
    total = np.array([0])
    for i in range(3):
        histSingle = cv2.calcHist([img], [i], None, [256], [0, 256])
        total = np.vstack((total, histSingle))
    return (total, pixal)


if __name__ == '__main__':
    start = time.time()
    # test_path = '/home/shichao/data/set/group4mean'
    # test_path = '/home/shichao/data/set/croppedgroup4mean'
    test_path = '/home/shichao/data/set/test'
    test_img_dirs = os.listdir(test_path)
    for test_img_dir in test_img_dirs:
        img_paths = glob.glob(os.path.join(os.path.join(test_path,test_img_dir),'*.*'))
        for img_path in img_paths:
            per_img_start = time.time()
            targetHist, targetPixal = get_histGBR(img_path)
            rootdir = '/home/shichao/Documents/training'
            # rootdir = '/home/shichao/data/set/group4mean'
            # rootdir = '/home/shichao/Documents/Image-classification/hist_lib'
            # rootdir = '/home/shichao/github/Image-classification/cropped_data_mean_train_31'
            # aHist = get_histGBR('a.png')
            # bHist = get_histGBR('Light.png')
            #
            # print(hist_similar(aHist, bHist))
            resultDict = {}
            for parent, dirnames, filenames in os.walk(rootdir):
                # for dirname in dirnames:
                #     print("parent is: " + parent)
                #     print("dirname is: " + dirname)
                for filename in filenames:
                    if (filename[-3:] == 'jpg'):
                        jpgPath = os.path.join(parent, filename)
                        testHist, testPixal = get_histGBR(jpgPath)
                        # print(hist_similar(targetHist,testHist)[0])
                        resultDict[jpgPath]=hist_similar(targetHist,testHist,targetPixal,testPixal)[0]

            # print(resultDict)
            # for each in resultDict:
            #     print(each, resultDict[each],sep="----")
            sortedDict = sorted(resultDict.items(), key=lambda asd: asd[1], reverse=True)
            print(test_img_dir+' '+os.path.split(os.path.split(sortedDict[0][0])[0])[1])
            # print(test_img_dir + ' ' + os.path.splitext(os.path.split(sortedDict[0][0])[1])[0])
            # for i in range(5):
            #     print(sortedDict[i])
            per_img_end = time.time()
            print('per img time is {0}'.format(per_img_end-per_img_start))
    end = time.time()
    print('elasped time is {0}'.format(end - start))
