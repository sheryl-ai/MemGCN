""" Code for loading data. """
import sklearn, sklearn.datasets
import sklearn.naive_bayes, sklearn.linear_model, sklearn.svm, sklearn.neighbors, sklearn.ensemble
import matplotlib.pyplot as plt
import scipy.sparse
import numpy as np
import time, re
import pickle as pkl
import hickle as hkl
import pandas as pd
import scipy.io as sio

from sklearn.model_selection import StratifiedKFold

bad_dti_id = [523, 524, 639, 643, 647, 767]

def load_dti(data_type):
    delete_sid = [373] + bad_dti_id
    subj = list()
    data = list()
    filepath = '../../../data/input/dti.roi/' + data_type
    sid = sio.loadmat(filepath + '_subject_id.mat')[data_type + '_subject_id'][0, :]
    for i in sid:
        if i in delete_sid:
            continue
        # print("reading connectivity file %s" % i)
        try:
            mat = sio.loadmat(filepath + '_' + str(i) + '.mat')['A']
            data.append(np.array(mat))
            subj.append(i)
            # print (mat)
        except IOError:
            print("File %s does not exit" % i)
    return data, subj

def load_roi_coords():
    f = open('../../../data/input/dti.coo.pd.pkl', 'rb')
    coords = pkl.load(f)
    return coords

def load_records(mem_size, code_size):
    f = open('../../../data/input/nonmotor.clinic.pkl', 'rb')
    print ('The memory size is: ', mem_size)
    print ('The clinical code dimension is: ', code_size)
    subject_array = pkl.load(f)
    n_samples = len(subject_array)
    records = list()
    print ('The number of sample with clinical records: ', n_samples)
    for i in range(n_samples):
        rec = np.zeros((mem_size, code_size), dtype=int)
        m = subject_array[i].shape[0] # m is length of sequence
        if m <= mem_size:
            rec[mem_size-m:mem_size, :] = subject_array[i]
        else:
            rec = subject_array[i][m-mem_size:, :]
        records.append(rec)
    records = np.array(records)
    return records

def load_data(data_type, mem_size, code_size):
    """Load data."""
    print (data_type)
    # load pairs
    f = open('../../../data/input/dti.pd.pairs.pkl', 'rb')
    pairs, labels = pkl.load(f)
    # print (len(pairs_labels))
    # pairs, labels = pairs_labels
    f.close()
    # print (len(pairs))
    # print (len(labels))

    # load roi coordinates
    coords = load_roi_coords()

    # load dti data
    data, subj = load_dti(data_type) # dictionary for multiview
    # print (len(data))
    # print (len(subj))
    data = np.array(data)

    # load clinical records
    records = load_records(mem_size, code_size)

    # train, validate, test split
    pairs = np.array(pairs)
    labels = np.array(labels)
    skf = StratifiedKFold(n_splits=5)
    pairs_set = list()
    labels_set = list()
    for train_index, test_index in skf.split(pairs, labels):
        train_x, test_x = pairs[train_index], pairs[test_index]
        train_y, test_y = labels[train_index], labels[test_index]
        val_x = test_x
        val_y = test_y
        pairs_set.append((train_x, val_x, test_x))
        labels_set.append((train_y, val_y, test_y))
    return data, subj, coords, records, pairs_set, labels_set
