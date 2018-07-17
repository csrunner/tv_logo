# -*- coding:utf-8 -*-
__author__ = 'shichao'



import numpy as np

def crop_image(img_seq):
    row = 3
    col = 3
    METHOD1 = False
    METHOD2 = True
    if len(img_seq.shape)==4:
        [dimx,dimy,dimc,dimt] = img_seq.shape
        cropped_img = np.zeros([2*dimx/row,2*dimy/col,dimc,dimt])
        for i in range(dimt):
            if METHOD1:
                cropped_img[:dimx/row,:dimy/col,:,i] = img_seq[:dimx/row,:dimy/col,:,i]
                cropped_img[:dimx/row,(2*dimy/col-dimy/col):,:,i] = img_seq[:dimx/row,(dimy-dimy/col):,:,i]
                cropped_img[(2*dimx/row-dimx/row):,:dimy/col,:,i] = img_seq[(dimx-dimx/row):,:dimy/col,:,i]
                cropped_img[(2*dimx/row-dimx/row):,(2*dimy/col-dimy/col):,:,i] = img_seq[(dimx-dimx/row):,(dimy-dimy/col):,:,i]
            if METHOD2:
                cropped_img[:dimx/row,:dimy/col,:,i]  = img_seq[:dimx/row,:dimy/col,:,i]
                cropped_img[:dimx/row,:-dimy/col,:,i] = img_seq[:dimx/row,:-dimy/col,:,i]
                cropped_img[:-dimx/row,:dimy/col,:,i] = img_seq[:-dimx/row,:dimy/col,:,i]
                cropped_img[:-dimx/row,:-dimy/col,:,i] = img_seq[:-dimx/row,:-dimy/col,:,i]
        return cropped_img

    else:
        [dimx,dimy,dimt] = img_seq.shape
        cropped_img = np.zeros([2*dimx/row,2*dimy/col,dimt])
        for i in range(dimt):
            if METHOD1:
                cropped_img[:dimx/row,:dimy/col,i] = img_seq[:dimx/row,:dimy/col,i]
                cropped_img[:dimx/row,(2*dimy/col-dimy/col):,i] = img_seq[:dimx/row,(dimy-dimy/col):,i]
                cropped_img[(2*dimx/row-dimx/row):,:dimy/col,i] = img_seq[(dimx-dimx/row):,:dimy/col,i]
                cropped_img[(2*dimx/row-dimx/row):,(2*dimy/col-dimy/col):,i] = img_seq[(dimx-dimx/row):,(dimy-dimy/col):,i]
            if METHOD2:
                cropped_img[:dimx/row,:dimy/col,i]  = img_seq[:dimx/row,:dimy/col,i]
                cropped_img[:dimx/row,-dimy/col:,i] = img_seq[:dimx/row,-dimy/col:,i]
                cropped_img[-dimx/row:,:dimy/col,i] = img_seq[-dimx/row:,:dimy/col,i]
                cropped_img[-dimx/row:,-dimy/col:,i] = img_seq[-dimx/row:,-dimy/col:,i]
        return cropped_img


