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
    from sklearn.feature_selection import SelectKBest, chi2
    from collections import Counter

    drug = sys.argv[1]
    num_threads = int(sys.argv[2])
    dataset = sys.argv[3]

    kmer_matrix = np.load("annotation/15mer_data/{}_kmer_matrix.npy".format(drug))
    kmer_rows_genomes = np.load("annotation/15mer_data/{}_kmer_rows.npy".format(drug))
    kmer_cols = np.load("annotation/15mer_data/{}_kmer_cols.npy".format(drug))

    # if using grdi genomes, this needs to change to the grdi dataframe
    mic_df = joblib.load("data/{}_mic_class_dataframe.pkl".format(dataset))
    mic_class_dict = joblib.load("data/{}_mic_class_order_dict.pkl".format(dataset))

    # replace genome names with matching mic value
    kmer_rows_mic = [mic_df[drug][i] for i in kmer_rows_genomes]
    num_starting_mic = len(kmer_rows_mic)

    # keep only rows with valid MIC labels
    assert(type(kmer_rows_mic[0])==type(mic_class_dict[drug][0]))
    row_mask = [i in mic_class_dict[drug] for i in kmer_rows_mic]
    kmer_matrix = kmer_matrix[row_mask]
    kmer_rows_mic = np.asarray(kmer_rows_mic)[row_mask]
    kmer_rows_genomes = np.asarray(kmer_rows_genomes)[row_mask]

    num_removed = num_starting_mic - np.sum(row_mask)
    if(num_removed>0):
        print("Removed {} invalid rows from {} kmer_matrix".format(num_removed,drug))

    # encode mic labels from 1,2,4,8 to 0,1,2,3
    le = preprocessing.LabelEncoder()
    le.classes_ = mic_class_dict[drug]
    y_train = le.transform(kmer_rows_mic)

    # calculate feature importances with anova f-value to compare against model importances
    import scipy.stats as stats
    anova = SelectKBest(chi2, k='all').fit(kmer_matrix,y_train)

    """
    f_vals = [] # to store the f-vals for each kmer

    # loop through columns of the data matrix
    for i, col in enumerate(kmer_matrix.T):
        class_lists = [] # this list stores a list of kmer counts for each class

        # loop through once per class
        for class_num in range(len(mic_class_dict[drug])):
            # return a list of only values matching class_num, and append that to our list of lists
            class_lists.append(list(col[[row_label == class_num for row_label in y_train]]))

        f_vals.append(stats.f_oneway(*class_lists).statistic)
    """

    num_classes_obj = len(Counter(y_train).keys())

    if(num_classes_obj==2):
        print("set objective to binary")
        objective = 'binary:logistic'
        other = 'multi:softmax'
    else:
        print("set objective to multiclass")
        objective = 'multi:softmax'
        other = 'binary:logistic'

    model = XGBClassifier(objective=objective, silent=True, nthread=num_threads)

    try:
        model.fit(kmer_matrix,y_train)
    except:
        print("UnExpected number of classes have data, switching objectives")
        model = XGBClassifier(objective=other, silent=True, nthread=num_threads)
        model.fit(kmer_matrix,y_train)

    feat_save = 'annotation/{}_{}feature_ranks.npy'.format(drug,dataset)
    np.save(feat_save, np.vstack((kmer_cols, model.feature_importances_,anova.scores_)))
