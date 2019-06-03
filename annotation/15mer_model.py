"""
Takes in an array of 15mers,
trains a model and then dumps the feature rankings
"""

if __name__ =="__main__":
    import numpy as np
    import pandas as pd
    from xgboost import XGBClassifier
    import sys, os
    from sklearn import preprocessing
    from sklearn.externals import joblib
    from collections import Counter

    drug = sys.argv[1]
    dataset = sys.argv[3]

    kmer_matrix = np.load("annotation/15mer_data/{}_kmer_matrix.npy".format(drug))
    kmer_rows_genomes = np.load("annotation/15mer_data/{}_kmer_rows.npy".format(drug))
    kmer_cols = np.load("annotation/15mer_data/{}_kmer_cols.npy".format(drug))

    # if using grdi genomes, this needs to change to the grdi dataframe
    mic_df = joblib.load("data/{}_mic_class_dataframe.pkl".format(dataset))
    mic_class_dict = joblib.load("data/{}_mic_class_order_dict.pkl".format(dataset))

    # replace genome names with matching mic value
    kmer_rows_mic = [mic_df[drug][i] for i in kmer_rows_genomes]

    # encode mic labels from 1,2,4,8 to 0,1,2,3
    le = preprocessing.LabelEncoder()
    le.classes_ = mic_class_dict[drug]
    y_train = le.transform(kmer_rows_mic)

    num_classes_obj = len(Counter(y_train).keys())

    if(num_classes_obj==2):
        print("set objective to binary")
        objective = 'binary:logistic'
        other = 'multi:softmax'
    else:
        print("set objective to multiclass")
        objective = 'multi:softmax'
        other = 'binary:logistic'

    num_threads = sys.argv[2]
    model = XGBClassifier(objective=objective, silent=True, nthread=num_threads)

    try:
        model.fit(x_train,y_train)
    except:
        print("UnExpected number of classes have data, switching objectives")
        model = XGBClassifier(objective=other, silent=True, nthread=num_threads)
        model.fit(x_train,y_train)

    feat_save = 'annotation/{}_{}feature_ranks.npy'.format(drug,dataset)
    np.save(feat_save, np.vstack((kmer_cols, model.feature_importances_)))
