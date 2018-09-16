w#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 21:02:52 2018

Name: khalednakhleh

Title: ECEN 689 Challenge 2
"""
import numpy as np
import pandas as pd
import de_idx as dx
from sklearn.linear_model import LogisticRegression as lr
from sklearn.neighbors import KNeighborsClassifier as knn
from timeit import default_timer as timer

"""----------------------------------------------------------------------------------------- """

def log_reg(x, y, t, q):
    
    # Logistic Regression predictor initialization 
    
    pred = lr(solver = "saga", max_iter = 200, multi_class = "multinomial", tol = 0.1)
    start = timer()                           # Start timer
    pred.fit(x,y)                             # Predictor training
    pred.result = pred.score(t,q)             # Predictor test
    pred.error = 1 - pred.result              # error probability
    pred.end = timer() - start                # End timer
    q = pred.predict(t)
    
    return q, pred
"""
def k_nn(x, y, t, q, n):
    
    k_pred = knn(n_neighbors = n, n_jobs = -1)
    k_pred.fit(x, y)
    k_pred.score(t, q)
    k_pred.predict(t)
    return k_pred

"""
def main():
    
    # user inputs file names in idx format
    training_set = input("\nEnter TRAINING SET file name: ") or "train-images-idx3-ubyte"
    training_label = input("\nEnter TRAINING LABEL file name: ") or "train-labels-idx1-ubyte"
    test_set = input("\nEnter TEST SET file name: ") or "t10k-images-idx3-ubyte"
    test_label = input("\nEnter TEST LABEL file name: ") or "t10k-labels-idx1-ubyte"
    k = input("\nEnter number of neighbors k [default is 5]: ") or 5

    x = dx.De_idx_set(training_set)     # Training images
    y = dx.De_idx_label(training_label) # Training labels
    
    t = dx.De_idx_set(test_set)         # Test images
    q = dx.De_idx_label(test_label)     # Test labels


    solution, predictor = log_reg(x.array, y.array, t.array, q.array)
    
    #k_class = k_nn(x.array, y.array, t.array, q.array, k)
    
    
    print("\nError percentage: " + str(round((predictor.error * 100), 5)) + " %")
    print("\nRun time: ", round(predictor.end, 5), " seconds.\n\n\n")
    print(solution.shape)
    
if __name__ == "__main__":
    
    main()
