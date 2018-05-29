# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import operator
import KNN
import pandas as pd

"""
Project: Handing Writing Classify
"""
def convert_Vect(filename):
    """
    convert handing writing data to vector
    """
    returnVect = np.zeros((1, 32*32))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr)
    return returnVect

filename = r'C:\Users\Administrator\Documents\数据挖掘常用算法\kNN算法\projects\手写数字识别系统\testDigits\0_0.txt'
print(convert_Vect(filename))


