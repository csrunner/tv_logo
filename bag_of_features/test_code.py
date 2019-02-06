# -*- coding:utf-8 -*-
__author__ = 'shichao'


import cv2
import numpy as np


image = cv2.imread("/Users/shichao/Downloads/20140506095721-494649420.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  #THRESH_OTSU
ret2, th2 = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY_INV) #GIVEN THRESH
print(ret1)
cv2.imshow('OTSU',th1)
cv2.waitKey(10000)

print(ret2)
cv2.imshow('GIVEN',th2)
cv2.waitKey(10000)
