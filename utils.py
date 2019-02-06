# -*- coding:utf-8 -*-
__author__ = 'shichao'



import numpy as np
import glob
import os
import cv2
import signal

def crop_image(img_seq):
    '''
    crop_image crop the raw image in the sequence frame by frame and gather the 4 corners into a new one
    :param img_seq:
    :return:
    '''
    row = 3
    col = 3
    METHOD1 = False
    METHOD2 = True
    if len(img_seq.shape)==4:
        [dimx,dimy,dimc,dimt] = img_seq.shape
        print(img_seq.shape)
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
        print(img_seq.shape)
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


def readin_image(readin_path,maxim_num=240,start_num=1):
    '''
    read in all images in one folder in the grey mode and stack the image into a sequence by the ascending order of Arabic
    :param readin_path:
    :param start_num: the first number of the image sequence, 0 or 1
    :return: the image stack and the flattened image stack
    '''
    img_paths = glob.glob(os.path.join(readin_path, '*.*'))
    img_vol = len(img_paths)

    img_vol = maxim_num
    filename = os.path.basename(img_paths[0])
    name,ext = os.path.splitext(filename)
    zfill_num = len(name)
    [dimx,dimy,_] = cv2.imread(img_paths[0]).shape
    flatten_img_stack = np.zeros([dimx * dimy, img_vol])
    img_stack = np.zeros([dimx, dimy, img_vol])

    for i in range(img_vol):
        img_path = os.path.join(readin_path,str(i+start_num).zfill(zfill_num)+ext)
        img_bgr = cv2.imread(img_path)
        img_stack[:,:,i] = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
        # print(i)
        # flatten_img_stack[:,i] = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY).flatten('C')

    return img_stack


def low_pass(flatten_array):
    '''
    get the low_pass signal of the image sequence
    :param flatten_array:
    :return:
    '''
    b, a = signal.butter(3, 0.08, 'low')
    # b, a = signal.cheby1(3,Wn=0.08,btype='low')

    low_pass_video_sig = np.zeros(flatten_array.shape)
    for i in range(flatten_array.shape[0]):

        sig_tmp = flatten_array[i,:]
        sig = []
        for j in range(len(sig_tmp)):
            sig.append(float(sig_tmp[j]))
        # print(sig)
        sf = signal.filtfilt(b, a, sig)
        low_pass_video_sig[i,:] = sf
    return low_pass_video_sig


def arbitrary_frame_diff(img_stack,step=1):
    '''
    get the sum of difference of two images with arbitrary steps
    :param flatten_img_stack:
    :param img_stack:
    :param step:
    :param use_flatten:
    :return:
    '''

    [dimx0, dimy0, dimz0] = img_stack.shape
    diff_stack = np.zeros([dimx0, dimy0, dimz0 - step])
    # previous = img_stack[:,:,0]
    for i in range(step, dimz0):
        this = img_stack[:, :, i]
        previous = img_stack[:, :, i - step]
        tmp = this - previous
        diff_stack[:, :, i - step] = abs(tmp)
        # print(i)

    # [dimx1,dimy1] = flatten_img_stack.shape
    # flatten_diff_stack = np.zeros([dimx1,dimy1-arbitrary_number])
    # # previous = np.zeros([dimx,arbitrary_number])
    # # # previous[:,0] = flatten_img_stack[:,0]
    # for i in range(arbitrary_number,dimy1):
    #     this = flatten_img_stack[:,i]
    #     previous = flatten_img_stack[:,i-arbitrary_number]
    #     flatten_img_stack[:,i-arbitrary_number] = abs(this-previous)
        # previous = this
    return diff_stack


def std(img_seq):
    '''
    get the standard deviation of the image sequence
    :param img_seq:
    :return:
    '''
    dimx,dimy,dimz = img_seq.shape
    img_sum = np.zeros([dimx,dimy])
    for i in range(dimz):
        img_sum += img_seq[:,:,i]
    img_mean = img_sum/dimz
    img_var = np.zeros([dimx,dimy])
    for i in range(dimz):
        img_var += np.power((img_seq[:,:,i]-img_mean),2)
    return np.sqrt(img_var)


def get_connected_components(bin_image,connectivity=4):
    '''
    Method: One component at a time connected component labeling
        Input:
    :param bin_image: Binary Image (h x w array of 0 and non-zero values, will created connected components of non-zero with 4 or 8 connectivity
    :return: labelled connected components ranging from 0 to counter-1
    '''
    h, w = bin_image.shape
    yc, xc = np.where(bin_image != 0)
    queue = []
    connected_array = np.zeros((h, w))  # labeling array
    counter = 1
    use_queue = True
    if use_queue:
        for elem in range(len(xc)):
            # iterate over all nonzero elements
            i = yc[elem]
            j = xc[elem]
            if connected_array[i, j] == 0:
                # not labeled yet proceed
                connected_array[i, j] = counter
                queue.append((i, j))
                while len(queue) != 0:
                    # work through queue
                    current = queue.pop(0)
                    i, j = current
                    if i == 0 and j == 0:
                        coords = np.array([[i, i + 1], [j + 1, j]])
                    elif i == h - 1 and j == w - 1:
                        coords = np.array([[i, i - 1], [j - 1, j]])
                    elif i == 0 and j == w - 1:
                        coords = np.array([[i, i + 1], [j - 1, j]])
                    elif i == h - 1 and j == 0:
                        coords = np.array([[i, i - 1], [j + 1, j]])
                    elif i == 0:
                        coords = np.array([[i, i, i + 1], [j - 1, j + 1, j]])
                    elif i == h - 1:
                        coords = np.array([[i, i, i - 1], [j - 1, j + 1, j]])
                    elif j == 0:
                        coords = np.array([[i, i + 1, i - 1], [j + 1, j, j]])
                    elif j == w - 1:
                        coords = np.array([[i, i + 1, i - 1], [j - 1, j, j]])
                    else:
                        if connectivity == 4:
                            coords = np.array([[i, i, i + 1, i - 1], [j - 1, j + 1, j, j]])
                        if connectivity == 8:
                            coords = np.array([[i, i, i + 1, i - 1, i - 1, i - 1, i + 1, i + 1],
                                               [j - 1, j + 1, j, j, j - 1, j + 1, j + 1, j - 1]])

                    for k in range(len(coords[0])):
                        # iterate over neighbor pixels, if  not labeled and not zero then assign current label
                        if connected_array[coords[0, k], coords[1, k]] == 0 and bin_image[coords[0, k], coords[1, k]] != 0:
                            connected_array[coords[0, k], coords[1, k]] = counter
                            queue.append((coords[0, k], coords[1, k]))
                counter += 1

    else:
        for elem in range(len(xc)):
            # iterate over all nonzero elements
            i = yc[elem]
            j = xc[elem]
            if connected_array[i, j] == 0:
                # not labeled yet proceed
                connected_array[i, j] = counter

                if i == 0 and j == 0:
                    coords = np.array([[i, i + 1], [j + 1, j]])
                elif i == h - 1 and j == w - 1:
                    coords = np.array([[i, i - 1], [j - 1, j]])
                elif i == 0 and j == w - 1:
                    coords = np.array([[i, i + 1], [j - 1, j]])
                elif i == h - 1 and j == 0:
                    coords = np.array([[i, i - 1], [j + 1, j]])
                elif i == 0:
                    coords = np.array([[i, i, i + 1], [j - 1, j + 1, j]])
                elif i == h - 1:
                    coords = np.array([[i, i, i - 1], [j - 1, j + 1, j]])
                elif j == 0:
                    coords = np.array([[i, i + 1, i - 1], [j + 1, j, j]])
                elif j == w - 1:
                    coords = np.array([[i, i + 1, i - 1], [j - 1, j, j]])
                else:
                    if connectivity == 4:
                        coords = np.array([[i, i, i + 1, i - 1], [j - 1, j + 1, j, j]])
                    if connectivity == 8:
                        coords = np.array([[i, i, i + 1, i - 1, i - 1, i - 1, i + 1, i + 1],
                                           [j - 1, j + 1, j, j, j - 1, j + 1, j + 1, j - 1]])

                for k in range(len(coords[0])):
                    # iterate over neighbor pixels, if  not labeled and not zero then assign current label
                    if connected_array[coords[0, k], coords[1, k]] == 0 and bin_image[coords[0, k], coords[1, k]] != 0:
                        connected_array[coords[0, k], coords[1, k]] = counter
                        queue.append((coords[0, k], coords[1, k]))
                counter += 1

    return connected_array,counter-1




