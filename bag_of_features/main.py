# -*- coding:utf-8 -*-
__author__ = 'shichao'

import os
import glob
import cv2
import numpy as np
from optparse import OptionParser
import time
from train import trainModel
from test import testModel
from utils import humanTime

def main():
    parser = OptionParser()
    parser.add_option("-p", "--testpath", dest="test_path", help="provide the test root path")
    parser.add_option("--resultpath", dest="result_path", help="provide the result path")
    (options, args) = parser.parse_args()
    test_path = options.test_path
    result_path = options.result_path

    train_start = time.time()
    trainModel()
    train_end = time.time()
    elasped_time = train_end-train_start
    print(humanTime(elasped_time))

    test_start = time.time()
    testModel(test_path,result_path)
    test_end = time.time()
    elasped_time = test_end-test_start
    print(humanTime(elasped_time))


if __name__ == "__main__":
    main()

