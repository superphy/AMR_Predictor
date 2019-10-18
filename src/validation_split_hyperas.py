#!/usr/bin/env python

"""
This Script to split the training set into quarters so that we have
3/5th train, 1/5th test, 1/5th validate
"""

import os, sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib
from sklearn import preprocessing

from model_evaluators import *
from data_transformers import *


if __name__=='__main__':
    dataset = sys.argv[1]
    drug = sys.argv[2]
    num_feats = sys.argv[3]
    kmer_length = sys.argv[4]

    # grdi dataset has no FIS classifications so we skip it
    if(dataset == 'grdi' and drug == 'FIS'):
        sys.exit()

    # pathings are public: 'AMP', grdi: 'grdi_AMP', kh: 'kh_AMP'
    if(dataset == 'public'):
        path = ''
    else:
        path = dataset+'_'

    # load the relevant data

    if(kmer_length == '11'):
        X = np.load(("data/filtered/{}{}/kmer_matrix.npy").format(path,drug))
        Y = np.load(("data/filtered/{}{}/kmer_rows_mic.npy").format(path,drug))
        Z = np.load(("data/filtered/{}{}/kmer_rows_genomes.npy").format(path,drug))
    else:
        X = np.load("data/multi-mer/kbest/{}/{}_{}_{}mer_matrix.npy".format(
        dataset, num_feats, drug, kmer_length))

        Z = np.load("data/multi-mer/{}{}mer_rows.npy".format(path, kmer_length))

        mic_df = joblib.load("data/{}_mic_class_dataframe.pkl".format(dataset))
        mic_class_dict = joblib.load("data/{}_mic_class_order_dict.pkl".format(dataset))

        Y = [mic_df[drug][i] for i in Z]

        row_mask = [i in mic_class_dict[drug] for i in Y]

        X = X[row_mask]
        Y = np.asarray(Y)[row_mask]
        Z = np.asarray(Z)[row_mask]



    Y = [remove_symbols(i) for i in Y]
    mic_class_dict = joblib.load("data/{}_mic_class_order_dict.pkl".format(dataset))
    mic_dict = [remove_symbols(i) for i in mic_class_dict[drug]]

    # possible label encodings are determined from all possible MIC values for that drug
    # for example, we change 0.25,0.5,1,2 into 0,1,2,3
    #le = preprocessing.LabelEncoder()
    #le.classes_ = mic_dict
    #Y = le.transform(Y)

    encoder = { mic_dict[i] : i for i in range(0, len(mic_dict))}
    Y = np.array([encoder[i] for i in Y])

    cv = StratifiedKFold(n_splits=5, random_state=913824)
    model_data = cv.split(X, Y, Z)

    set_count = 0
    for train, test in model_data:
        set_count+=1
        x_train = X[train]
        x_test  = X[test]
        y_test  = Y[test]
        y_train = Y[train]
        z_train = Z[train]
        z_test  = Z[test]

        if kmer_length != '11':
            kmer = '_'+kmer_length+'mer'
        else:
            kmer = ''

        if not os.path.exists(os.path.abspath(os.path.curdir)+"/data/filtered/{}{}{}/splits/set".format(path,drug,kmer)+str(set_count)):
            os.makedirs(os.path.abspath(os.path.curdir)+"/data/filtered/{}{}{}/splits/set".format(path,drug,kmer)+str(set_count))


        save_path = "data/filtered/{}{}{}/splits/set".format(path,drug,kmer)+str(set_count)

        # This just saves the testing set, so the data is split into 5ths, each set is 1/5th of the data
        # x is the 2D matrix of kmer counts
        # y is the row labels as MIC values
        # z is the row labels as genome ID's

        np.save(save_path+'/x.npy', x_test)
        np.save(save_path+'/y.npy', y_test)
        np.save(save_path+'/z.npy', z_test)
