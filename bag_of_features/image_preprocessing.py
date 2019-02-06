# -*- coding:utf-8 -*-
__author__ = 'shichao'

import numpy as np
import os
import cv2
import glob
from optparse import OptionParser


def crop_image(input_root, output_root, row_cut, collom_cut):
    root = input_root
    dirnames = os.listdir(root)
    for dirname in dirnames:
        sub_directory = os.path.join(root, dirname)
        filenames = glob.glob(os.path.join(sub_directory, '*.*'))
        for filename in filenames:
            img = cv2.imread(filename)
            img_name = os.path.split(filename)[1]
            [dim1,dim2,dim3] = img.shape
            cropped_img = img[:dim1/row_cut, :dim2/collom_cut, dim3]
            output_directory = os.path.join(output_root, dirname)
            if not os.path.exists(output_directory):
                os.mkdir(output_directory)
            cv2.imwrite(os.path.join(output_directory, img_name), cropped_img)



if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-p", "--inpath", dest="input", help="given the input path")
    parser.add_option("--outpath", dest="output", help="given the output path")
    (options, args) = parser.parse_args()

    input_root = options.input
    output_root = options.output
    row_cut = 3
    collom_cut = 4
    crop_image(input_root, output_root, row_cut, collom_cut)