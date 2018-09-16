#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 17:48:01 2018

Name: khalednakhleh
"""

import struct as st
import numpy as np
from sklearn.linear_model import LogisticRegression as lr
from timeit import default_timer as timer

def log_reg(x, y, t, q):
    
    # Logistic Regression predictor initialization 
    predictor = lr(solver = "lbfgs")
    
    start = timer()                           # Start timer
    predictor.fit(x, y)                       # Predictor training
    result = predictor.score(t, q)            # Predictor test
    
    end = timer() - start                     # End timer
    error = 1 - result
    
    return error, end

def de_idx_set(file_name):
    file = open(file_name, "rb")
    
    file.seek(0)
    magic_no = st.unpack('>4B', file.read(4))
    item_no = st.unpack('>I', file.read(4))[0]
    row_no = st.unpack('>I', file.read(4))[0]
    col_no = st.unpack('>I', file.read(4))[0]

    bytes_no = 1 * item_no * row_no * col_no
    dimension = (item_no, col_no * row_no)
    array = (255 - np.asarray(st.unpack('>' + 'B' * bytes_no,
             file.read(bytes_no))).reshape(dimension))

    return array


def de_idx_label(file_name):
    file = open(file_name, "rb")
    
    file.seek(0)
    magic_no = st.unpack('>4B', file.read(4))
    label_no = st.unpack('>I', file.read(4))[0]


    array = 255 - np.asarray(st.unpack('>' + 'B' * label_no,
            (file).read(label_no)))

    return array

x = de_idx_set("train-images-idx3-ubyte")    # Training images
y = de_idx_label("train-labels-idx1-ubyte")  # Training labels

t = de_idx_set("t10k-images-idx3-ubyte")     # Test images
q = de_idx_label("t10k-labels-idx1-ubyte")   # Test labels

error, end = log_reg(x, y, t, q)


print("\nError percentage: " + str(round((error * 100), 5)) + " %.")
print("\nRun time: " + str(round(end, 5)) + " seconds.")
