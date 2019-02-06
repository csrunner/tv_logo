# -*- coding:utf-8 -*-
__author__ = 'shichao'

import cv2
import os

def video_frame(video_path,frames_path):
    vc = cv2.VideoCapture(video_path)  # 读入视频文件
    c = 0

    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        rval = False

    timeF = 1  # 视频帧计数间隔频率

    while rval:  # 循环读取视频帧
        rval, frame = vc.read()
        if (c % timeF == 0):  # 每隔timeF帧进行存储操作
            cv2.imwrite(os.path.join(frames_path,str(c).zfill(5) + '.jpg'), frame)  # 存储为图像
        c = c + 1
        # cv2.waitKey(1)
    vc.release()


video_path = '/Users/shichao/Downloads/VID_20180913_171446.mp4'
frame_path = '/Users/shichao/Downloads/'
frame_path = '/Users/shichao/working_dir/data/'
frame_path = '/Users/shichao/workding_dir/data'
def main():
    video_frame(video_path,frame_path)

if __name__ == '__main__':
    main()