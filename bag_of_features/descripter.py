# -*- coding:utf-8 -*-
__author__ = 'shichao'


import cv2
import numpy as np
import config


def surf(img,y,class_number):
    '''

    :param img:
    :param y:
    :param class_number:
    :return:
    '''
    img_descs = []
    surf_feature = cv2.xfeatures2d.SURF_create(hessianThreshold=config.SURF_HESSIANTHRESHOLD,nOctaves=config.OCTAVES,\
        nOctaveLayers=config.OCTAVELAYERS)
    kp, des = surf_feature.detectAndCompute(img,None)
    if des is not None:
        img_descs.append(des)
        y.append(class_number)
    return img_descs,y


def img_to_vect(img_descs,cluster_model):
    '''
    get the BoV
    :param img_descs:
    :param cluster_model:
    :return:
    '''

    clustered_desc = [cluster_model.predict(raw_words) for raw_words in img_descs]
    img_bov_hist = np.array(np.bincount(clustered_desc,minlength=cluster_model.n_clusters))
    return img_bov_hist