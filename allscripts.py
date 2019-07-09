#All scripts
from sklearn import metrics
import os
import sys

from pandas import DataFrame
from IPython.display import HTML

import numpy as np
import pandas as pd

from fastai.basic_data import *
from fastai.basic_train import *
from fastai.callbacks import *
from fastai.data_block import *
from fastai.metrics import *
from fastai.train import *
from fastai.utils import *
from fastai.core import *
from fastai.gen_doc import *

#from fastai import Learner,DataBunch

import torch
import torch.nn as nn
import torch.utils.data as tdatautils

import glob
import re

import time
import datetime

from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import average_precision_score
from sklearn import random_projection
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
def norm3d(df):
    train_data = df
    scaler = StandardScaler()
    num_instances, num_time_steps, num_features = train_data.shape
    train_data = np.reshape(train_data, newshape=(-1, num_features))
    train_data = scaler.fit_transform(train_data)
    train_data = np.reshape(train_data, newshape=(num_instances, num_time_steps, num_features))
    return train_data

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

#Upsampler
def upsample(X, y):
    Xp = []
    yp = []

    threshold = 0.5

    neg_index = np.where(y == 0)[0]
    pos_index = np.where(y == 1)[0]

    choices_neg = list(neg_index)
    choices_pos = list(pos_index)

    while len(choices_neg) > 0:
        chance = np.random.rand()

        if chance > threshold:
            choice = np.random.choice(choices_pos)
            choices_pos.remove(choice)
            if len(choices_pos) == 0:
                choices_pos = list(pos_index)

            y_value = 1
            #print(i, choice, "pos")

        if chance <= threshold:
            choice = np.random.choice(choices_neg)
            choices_neg.remove(choice)
            y_value = 0
            #print(i, choice, "neg")

        Xp.append(X[choice])
        yp.append(y_value)

        #if len(yp) % 100 == 99:
        #    print(len(yp))

    # Create numpy array
    dim1 = len(Xp)
    dim2 = Xp[0].shape[0]
    dim3 = Xp[0].shape[1]

    df = np.zeros(shape = (dim1,dim2,dim3))
    
    for i in range(0, dim1):
        df[i] = Xp[i]
        
    Xp = df
    yp = np.array(yp)
    print(Xp.shape, yp.shape)
    return(Xp, yp)
    
# Random projection module
def generate_weights(batch_size, n_hid, new):
    if new == 1:
        print("Creating new m1, m2")
        #Gaussian noise
        batch_size = batch_size
        n_hid = n_hid
        # Final maxpool layer default gives size 32 * 20 = 640. With additional maxpool = 16 * 20 = 320.
        bs = batch_size
        n_hid = int(n_hid)

        #bs 32, N-hid 20 = 640
        # w1.shape = 32, 320
        # m1.shape = 32, 640

        #Create pre-set gaussian noise vector
        w1 = np.random.normal(size = (int(1), int(n_hid))).astype(np.float32)
        w2 = np.random.normal(size = (int(1), int(n_hid))).astype(np.float32)

        transformer = Normalizer().fit(w1)
        w1 = transformer.transform(w1)
        #w1 = w1.reshape(int(bs) * int(n_hid/2))
        w1 = np.vstack([w1] * int(bs/2))

        transformer = Normalizer().fit(w2)
        w2 = transformer.transform(w2)
        #w2 = w2.reshape(int(bs) * int(n_hid/2))
        w2 = np.vstack([w2] * int(bs/2))
        print(w2.shape)

        m1 = torch.from_numpy(np.append(w1,w2)).view(bs, n_hid).cuda()
        m2 = torch.from_numpy(np.append(w2,w1)).view(bs, n_hid).cuda()

        np.save("/home/maghoi/main/data/m1.npy", m1)
        np.save("/home/maghoi/main/data/m2.npy", m2)
        
    elif glob.glob("/home/maghoi/main/data/m1.npy") and glob.glob("/home/maghoi/main/data/m2.npy"):
        print("Loading m1, m2")
        m1 = torch.from_numpy(np.load("/home/maghoi/main/data/m1.npy")).cuda()
        m2 = torch.from_numpy(np.load("/home/maghoi/main/data/m2.npy")).cuda()

    print(m1.shape, m2.shape)
    return(m1, m2)

### Start position 
def data_generator(train, valid, test):
    filelist = train + valid + test; len(filelist)

    data_size = len(filelist)
    ix_val = len(train)
    ix_test = len(train) + len(valid)

    filelist_loaded = []

    #Load data into dfs
    for i in range(0, len(filelist)):
        df = np.load(filelist[i])
        filelist_loaded.append(df)

    #Initialize empty df ordered by complexes and aminos
    dim1 = range(0, data_size)
    dim2 = filelist_loaded[0].shape[0]
    dim3 = filelist_loaded[0].shape[1]
    x = np.zeros(shape = (data_size, dim2, dim3))

    for i in range(0, data_size):
        x[i] = filelist_loaded[i]

    #Extract y
    y = np.zeros(shape = (data_size), dtype="int64")

    counter_x = range(0, data_size)
    counter_y = range(len(y))
    for c_x, c_y in zip(counter_x, counter_y):
        r = re.compile(r'.*P1.*')
        if bool(r.match(filelist[c_x])):
            y[c_y] = 1

    X_train, y_train = x[0 : ix_val], y[0 : ix_val]
    X_val, y_val = x[ix_val : ix_test], y[ix_val : ix_test]
    X_test, y_test = x[ix_test : ], y[ix_test : ]
    return X_train, y_train, X_val, y_val, X_test, y_test


def data_generator_blosum(X_mhc, X_pep, X_tcr_a, X_tcr_b, target_y, names, PAR_VEC, data_size=1098, norm = False, val_part = 4):
    filelist = names; len(filelist)
    
    #set validation part
    partitions = [0, 1, 2, 3, 4]
    partitions.remove(val_part)

    train_part = partitions.copy()
    train = train_part
    valid = [val_part]
    test = [val_part]

    filelist_loaded = []
    for i in range(0, len(filelist)):
        df = np.vstack((X_mhc[i], X_pep[i], X_tcr_a[i], X_tcr_b[i]))
        filelist_loaded.append(df)

    dim1 = range(0, data_size)
    dim2 = filelist_loaded[0].shape[0]
    dim3 = filelist_loaded[0].shape[1]
    x = np.zeros(shape = (data_size, dim2, dim3))

    for i in range(0, data_size):
        x[i] = filelist_loaded[i]
        
    #Norm
    if norm == True:
        print("Normalizing with sci-kit L2 norm")
        x = norm3d(x)
    print("Train part", train_part)
    print("Val part", val_part)

    y = target_y[:, 1]
    train_idx = np.where(np.array((PAR_VEC == train[0]) + (PAR_VEC == train[1]) + (PAR_VEC == train[2]) + PAR_VEC == train[3]))
    valid_idx = np.where(np.array(PAR_VEC) == valid[0])
    test_idx = np.where(np.array(PAR_VEC) == valid[0])
    
    #y = target_y[:, 1]
    #train_idx = np.where(np.array((PAR_VEC == 0) + (PAR_VEC == 1) + (PAR_VEC == 2) + PAR_VEC == 3))
    #valid_idx = np.where(np.array(PAR_VEC) == 4)
    #test_idx = np.where(np.array(PAR_VEC) == 4)

    X_train, X_valid, X_test = x[train_idx], x[valid_idx], x[test_idx]
    y_train, y_valid, y_test = y[train_idx], y[valid_idx], y[test_idx]
    return(X_train, y_train, X_valid, y_valid, X_test, y_test)

### Start position 0
def data_generator_filenames(ix_train = 1080, ix_val = 256, ix_test = 128, data_size=1464):
    #filelist = glob.glob("/scratch/maghoi/pMHC_data/features8/*.npy"); len(filelist)
    filelist = glob.glob("/home/maghoi/pMHC_data/features6/*.npy"); len(filelist)

    filelist_loaded = []
    ix_test = data_size - ix_test
    ix_val = ix_test - ix_val
    ix_train = ix_val - ix_train

    #Load data into dfs
    for i in range(0, len(filelist)):
        df = np.load(filelist[i])
        filelist_loaded.append(df)

    #Initialize empty df ordered by complexes and aminos
    dim1 = range(0, data_size)
    dim2 = filelist_loaded[0].shape[0]
    dim3 = filelist_loaded[0].shape[1]
    x = np.zeros(shape = (data_size, dim2, dim3))

    for i in range(0, data_size):
        x[i] = filelist_loaded[i]

    #Extract y
    y = np.zeros(shape = (data_size), dtype="int64")

    counter_x = range(0, data_size)
    counter_y = range(len(y))
    for c_x, c_y in zip(counter_x, counter_y):
        r = re.compile(r'.*P1.*')
        if bool(r.match(filelist[c_x])):
            y[c_y] = 1

    X_train, y_train = x[0 : ix_val], y[0 : ix_val]
    X_val, y_val = x[ix_val : ix_test], y[ix_val : ix_test]
    X_test, y_test = x[ix_test : ], y[ix_test : ]
    return X_train, y_train, X_val, y_val, X_test, y_test

