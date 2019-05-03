# -*- coding:utf-8 -*-
__author__ = 'shichao'

from tensorflow import keras
from collections import Counter
import glob

threshold = 0.8

def detection(raw_img):
    left_top_img = raw_img[:,:,:]
    det,confidence = keras.fit(left_top_img)

    if confidence < threshold:
        right_top_img = raw_img[:,:,:]
        det,confidence = keras.fit(right_top_img)

        if confidence < threshold:
            left_bottom_img = raw_img[:,:,:]
            det,confidence = keras.fit(left_bottom_img)

            if confidence < threshold:
                right_bottom_img = raw_img[:,:,:]
                det,confidence = keras.fit(right_bottom_img)

    return det,confidence

def fusion(img_list):
    det_list = []
    confidence_list = []

    for img in img_list:
        det_one_frame,confidence_one_frame = detection(img)
        det_list.append(det_one_frame)
        confidence_list.append(confidence_one_frame)
        count = Counter(det_list)
        ratio = count.most_common()[0][1]/float(count.most_common()[1][1])
        if len(det_list) > 10 and ratio > 1.5:
            return count.most_common()[0][0]
    return 0

def main():

    img_list = glob.glob('')
    video_result = fusion(img_list)
    print(video_result)

if __name__ == '__main__':
    main()


