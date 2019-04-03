# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:18:22 2018

@author: Thanh Tung Khuat

Preprocessing functions helper

"""

import numpy as np
import itertools
from functionhelper.bunchdatatype import Bunch

dtype = np.float64

def normalize(A, new_range, old_range = None):
    """
    Normalize the input dataset
    
    INPUT
        A           Original dataset (numpy array) [rows are samples, cols are features]
        new_range   The range of data after normalizing
        old_range   The old range of data before normalizing
   
    OUTPUT
        Normalized dataset
    """
    D = A.copy()
    n, m = D.shape
    
    for i in range(m):
        v = D[:, i]
        if old_range is None:
            minv = np.nanmin(v)
            maxv = np.nanmax(v)
        else:
            minv = old_range[0]
            maxv = old_range[1]
        
        if minv == maxv:
            v = np.ones(n) * 0.5;
        else:      
            v = new_range[0] + (new_range[1] - new_range[0]) * (v - minv) / (maxv - minv)
        
        D[:, i] = v;
    
    return D
    

def load_dataset(path, percentTr, isNorm = False, new_range = [0, 1], old_range = None, class_col = -1):
    """
    Load file containing dataset and convert data in the file to training and testing datasets. Class labels are located in the last column in the file
    Note: Missing value in the input file must be question sign ?
    
        Xtr, Xtest, patClassIdTr, patClassIdTest = load_dataset(path, percentTr, True, [0, 1])
    
    INPUT
       path             the path to the data file (including file name)
       percentTr        the percentage of data used for training (0 <= percentTr <= 1)
       isNorm           identify whether normalizing datasets or not, True => Normalized
       new_range        new range of datasets after normalization
       old_range        the range of original datasets before normalization (all features use the same range)
       class_col        -1: the class label is the last column in the dataset
                        otherwise: the class label is the first column in the dataset

    OUTPUT
       Xtr              Training dataset
       Xtest            Testing dataset
       patClassIdTr     Training class labels
       patClassIdTest   Testing class labels
       
    """
    
    lstData = []
    with open(path) as f:
        for line in f:
            nums = np.fromstring(line.rstrip('\n').replace(',', ' ').replace('?', 'nan'), dtype=dtype, sep=' ').tolist()
            if len(nums) > 0:
                lstData.append(nums)
#            if (a.size == 0):
#                a = nums.reshape(1, -1)
#            else:
#                a = np.concatenate((a, nums.reshape(1, -1)), axis=0)
    A = np.array(lstData, dtype=dtype)
    YA, XA = A.shape
    
    if class_col == -1:
        X_data = A[:, 0:XA-1]
        classId_dat = A[:, -1]
    else:
        # class label is the first column
        X_data = A[:, 1:]
        classId_dat = A[:, 0]
        
    classLabels = np.unique(classId_dat)
    
    # class labels must start from 1, class label = 0 means no label
    if classLabels.size > 1 and np.size(np.nonzero(classId_dat < 1)) > 0:
        classId_dat = classId_dat + 1 + np.min(classId_dat)
        classLabels = classLabels + 1 + np.min(classLabels)

    if isNorm:
        X_data = normalize(X_data, new_range, old_range)
    
    if percentTr != 1 and percentTr != 0:
        noClasses = classLabels.size
        
        Xtr = np.empty((0, XA - 1), dtype=dtype)
        Xtest = np.empty((0, XA - 1), dtype=dtype)

        patClassIdTr = np.array([], dtype=np.int64)
        patClassIdTest = np.array([], dtype=np.int64)
    
        for k in range(noClasses):
            idx = np.nonzero(classId_dat == classLabels[k])[0]
            # randomly shuffle indices of elements belonging to class classLabels[k]
            if percentTr != 1 and percentTr != 0:
                idx = idx[np.random.permutation(len(idx))] 
    
            noTrain = int(len(idx) * percentTr + 0.5)
    
            # Attach data of class k to corresponding datasets
            Xtr_tmp = X_data[idx[0:noTrain], :]
            Xtr = np.concatenate((Xtr, Xtr_tmp), axis=0)
            patClassId_tmp = np.full(noTrain, classLabels[k], dtype=np.int64)
            patClassIdTr = np.append(patClassIdTr, patClassId_tmp)
            
            patClassId_tmp = np.full(len(idx) - noTrain, classLabels[k], dtype=np.int64)
            Xtest = np.concatenate((Xtest, X_data[idx[noTrain:len(idx)], :]), axis=0)
            patClassIdTest = np.concatenate((patClassIdTest, patClassId_tmp))
        
    else:
        if percentTr == 1:
            Xtr = X_data
            patClassIdTr = np.array(classId_dat, dtype=np.int64)
            Xtest = np.array([])
            patClassIdTest = np.array([])
        else:
            Xtr = np.array([])
            patClassIdTr = np.array([])
            Xtest = X_data
            patClassIdTest = np.array(classId_dat, dtype=np.int64)
        
    return (Xtr, Xtest, patClassIdTr, patClassIdTest)


def load_dataset_without_class_label(path, percentTr, isNorm = False, new_range = [0, 1]):
    """
    Load file containing dataset without class label and convert data in the file to training and testing datasets.
    
        Xtr, Xtest = load_dataset_without_class_label(path, percentTr, True, [0, 1])
    
    INPUT
       path             the path to the data file (including file name)
       percentTr        the percentage of data used for training (0 <= percentTr <= 1)
       isNorm           identify whether normalizing datasets or not, True => Normalized
       new_range        new range of datasets after normalization

    OUTPUT
       Xtr              Training dataset
       Xtest            Testing dataset
       
    """
    lstData = []
    with open(path) as f:
        for line in f:
            nums = np.fromstring(line.rstrip('\n').replace(',', ' '), dtype=dtype, sep=' ').tolist()
            if len(nums) > 0:
                lstData.append(nums)
#            if (X_data.size == 0):
#                X_data = nums.reshape(1, -1)
#            else:
#                X_data = np.concatenate((X_data, nums.reshape(1, -1)), axis = 0)
    X_data = np.array(lstData, dtype=dtype)
    if isNorm:
        X_data = normalize(X_data, new_range)
        
    # randomly shuffle indices of elements in the dataset
    numSamples = X_data.shape[0]
    newInds = np.random.permutation(numSamples)
    
    if percentTr != 1 and percentTr != 0:
        noTrain = int(numSamples * percentTr + 0.5)
        Xtr = X_data[newInds[0:noTrain], :]
        Xtest = X_data[newInds[noTrain:], :]
    else:
        if percentTr == 1:
            Xtr = X_data
            Xtest = np.array([])
        else:
            Xtr = np.array([])
            Xtest = X_data
        
    return (Xtr, Xtest)


def save_data_to_file(path, X_data):
    """
    Save data to file
    
    INPUT
        path        The path to the data file (including file name)
        X_data      The data need to be stored
    """
    np.savetxt(path, X_data, fmt='%f', delimiter=', ')   
    

def string_to_boolean(st):
    if st == "True" or st == "true":
        return True
    elif st == "False" or st == "false":
        return False
    else:
        raise ValueError