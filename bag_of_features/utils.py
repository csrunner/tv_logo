# -*- coding:utf-8 -*-
__author__ = 'shichao'

import cv2
import glob
import numpy as np
import os
import config
import time
import matplotlib.pyplot as plt

'''
this script includes the functions
calculate IOU by the func calc_IOU
display the elasped time in the form hour:minute:second by the func humanTime
display image keypoints detected by surf by the func displayKeyPoints

'''
def bbx():
    pass

def plot_img_hist(img):
    if (len(img.shape)==2):
        plt.hist(img.ravel(), 256, [0, 256])
        plt.show()
    if (len(img.shape)==3):
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.show()


def calc_IOU(box1, box2):
    '''
    this function calculate the IOU from the given coordinates of box1 and box2
    :param box1: the coordinate of box in the form (x1,y1,x2,y2)
    :param box2: the coordinate of box in the form (x1,y1,x2,y2)
    :return:
    '''
    x1 = max(box1[0],box2[0])
    y1 = max(box1[1],box2[1])
    x2 = min(box1[2],box2[2])
    y2 = min(box1[3],box2[3])
    if (x1>x2 or y1>y2):
        print('no intersection')
        return 0
    intersection = float((x2-x1)*(y2-y1))
    union = abs((box1[2]-box1[0])*(box1[3]-box1[1])) + abs((box2[2]-box2[0])*(box2[3]-box2[1]))
    IOU = intersection/union
    return IOU

def humanTime(sec):
    '''
    print the elasped time in the form of day:hour:minute:second
    :param sec:
    :return:
    '''
    mins, secs = divmod(sec,60)
    hours, mins = divmod(mins,60)
    days, hours = divmod(hours,24)

    return '%02d:%02d:%02d:02f'%(days,hours,mins,secs)


def displayKeyPoints(dirname,outpath):
    '''
    display keypoints detected by the descriptors
    :param dirname: the direcotry contains images to be detected
    :param outpath: the direcotry to save the featured images
    :return:
    '''
    img_paths = glob.glob(os.path.join(dirname,'*.*'))
    # img = (cv2.imread(img_path) for img_path in img_paths)
    for img_path in img_paths:
        img = cv2.imread(img_path)
        filename = os.path.split(img_path)[1]
        surf_feature = cv2.xfeatures2d.SURF_create(hessianThreshold=config.SURF_HESSIANTHRESHOLD)
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        dst = np.zeros(gray.shape)
        kp, des = surf_feature.detectAndCompute(gray,None)
        cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        cv2.imwrite(os.path.join(outpath,filename),img)



'''-------image histogram match---------'''
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


def img_hist_similarity(test_path, rootdir):
    start = time.time()
    result = list()
    # test_path = '/home/shichao/data/set/group4mean'
    # test_path = '/home/shichao/data/set/croppedgroup4mean'
    # test_path = '/home/shichao/data/set/test'
    test_img_dirs = os.listdir(test_path)
    for test_img_dir in test_img_dirs:
        img_paths = glob.glob(os.path.join(os.path.join(test_path, test_img_dir), '*.*'))
        for img_path in img_paths:
            per_img_start = time.time()
            targetHist, targetPixal = get_histGBR(img_path)
            # rootdir = '/home/shichao/Documents/training'
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
                    if (filename[-3:] == 'jpg' or 'jpeg'):
                        jpgPath = os.path.join(parent, filename)
                        testHist, testPixal = get_histGBR(jpgPath)
                        # print(hist_similar(targetHist,testHist)[0])
                        resultDict[jpgPath] = hist_similar(targetHist, testHist, targetPixal, testPixal)[0]

            # print(resultDict)
            # for each in resultDict:
            #     print(each, resultDict[each],sep="----")
            sortedDict = sorted(resultDict.items(), key=lambda asd: asd[1], reverse=True)
            # print(test_img_dir + ' ' + os.path.split(os.path.split(sortedDict[0][0])[0])[1])
            result.append(test_img_dir + ' ' + os.path.split(os.path.split(sortedDict[0][0])[0])[1])
            # print(test_img_dir + ' ' + os.path.splitext(os.path.split(sortedDict[0][0])[1])[0])
            # for i in range(5):
            #     print(sortedDict[i])
            per_img_end = time.time()
            print('per img time is {0}'.format(per_img_end - per_img_start))
    end = time.time()
    print('elasped time is {0}'.format(end - start))
    # print(result)
    for output in result:
        print(output)

'''-----------------------end of image hist match module-----------------------------------------'''
