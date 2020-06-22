"""
Takes in an array of 15mers,
trains a model and then dumps the feature rankings

Also saves feature selected matrices along the way
"""


"""
If you dont force class and pass 10000 features, each splits gets 10k features and the final model is the best 10k of the total 10k*10 splits
If you do pass force class feats and pass 10000, each class of each split gets 10000 features, all are kept, no furthur feature selection
^^^ can rewrite to merge each class matrix, do feature selection, and then merge the rest of the matrix.
"""


if __name__ =="__main__":
    import numpy as np
    import pandas as pd
    from xgboost import XGBClassifier
    import sys, os
    from sklearn import preprocessing
    from sklearn.externals import joblib
    from sklearn.feature_selection import SelectKBest, chi2, f_classif
    from collections import Counter
    import psutil
    
    drug = sys.argv[1]
    num_threads = int(sys.argv[2])
    dataset = sys.argv[3]
    kmer_length = sys.argv[4]
    num_feats = sys.argv[5]
    print(drug,dataset)
    # if we want the top X features predictive of each class, not just overall
    force_per_class = int(sys.argv[6])

    if dataset == 'public':
        dataset_path = ''
    else:
        dataset_path = 'grdi_'

    kmer_matrix = np.load("data/multi-mer/{}{}mer_matrix.npy".format(dataset_path, kmer_length))
    print("kmer_matrix:", sys.getsizeof(kmer_matrix))
    print(psutil.virtual_memory())
    kmer_rows_genomes = np.load("data/multi-mer/{}{}mer_rows.npy".format(dataset_path, kmer_length))
    kmer_cols = np.load("data/multi-mer/{}{}mer_cols.npy".format(dataset_path, kmer_length))

    # if using grdi genomes, this needs to change to the grdi dataframe
    mic_df = joblib.load("data/{}_mic_class_dataframe.pkl".format(dataset))
    mic_class_dict = joblib.load("data/{}_mic_class_order_dict.pkl".format(dataset))

    # replace genome names with matching mic value
    kmer_rows_mic = [mic_df[drug][i] for i in kmer_rows_genomes]
    num_starting_mic = len(kmer_rows_mic)

    # keep only rows with valid MIC labels
    def to_str(i):
        if isinstance(i, str):
            return i
        else:
            return str(i)
    print('rows_mic[0]:',type(kmer_rows_mic[0]), kmer_rows_mic[0])
    print('class_dict[0]:',type(mic_class_dict[drug][0]), mic_class_dict[drug][0])
    kmer_rows_mic = [to_str(i) for i in kmer_rows_mic]
    assert(type(kmer_rows_mic[0])==type(mic_class_dict[drug][0]))    
    row_mask = [i in mic_class_dict[drug] for i in kmer_rows_mic]
    kmer_matrix = kmer_matrix[row_mask]
    kmer_rows_mic = np.asarray(kmer_rows_mic)[row_mask]
    kmer_rows_genomes = np.asarray(kmer_rows_genomes)[row_mask]

    print("kmer_matrix:", sys.getsizeof(kmer_matrix))
    print(psutil.virtual_memory())

    num_removed = num_starting_mic - np.sum(row_mask)
    if(num_removed>0):
        print("Removed {} invalid rows from {} kmer_matrix".format(num_removed,drug))

    # encode mic labels from 1,2,4,8 to 0,1,2,3
    #le = preprocessing.LabelEncoder()
    #le.classes_ = mic_class_dict[drug]
    #y_train = le.transform(kmer_rows_mic)

    # encode mic labels from 1,2,4,8 to 0,1,2,3
    encoder = { mic_class_dict[drug][i] : i for i in range(0, len(mic_class_dict[drug]))}
    y_train = [encoder[i] for i in kmer_rows_mic]

    # feature selection with anova f val to the top 1000 features
    import gc
    gc.collect()

    """
    if matrix has been split, use that, if not, split it
    """
    matrix_path = "/data/multi-mer/kbest/{}/{}_{}_{}mer_matrix.npy".format(
    dataset, num_feats, drug, kmer_length)

    if not os.path.exists(os.path.abspath(os.path.curdir)+matrix_path) or force_per_class == 1:
            step_size = int(len(kmer_cols) / 10)
            split_starts = [step_size*i for i in range(11)]

            # save each split, run normally for force_per_class
            if not os.path.exists(os.path.abspath(os.path.curdir)+"/data/multi-mer/kbest/{}/all_{}_{}mer_matrix{}.npy".format(dataset, drug, kmer_length, 1)):
                for i in range(10):
                    np.save("data/multi-mer/kbest/{}/all_{}_{}mer_matrix{}.npy".format(
                    dataset, drug, kmer_length, i+1), ((kmer_matrix.T)[split_starts[i]:split_starts[i+1]]).T)
                    np.save("data/multi-mer/kbest/{}/all_{}_{}mer_cols{}.npy".format(
                    dataset, drug, kmer_length, i+1), kmer_cols[split_starts[i]:split_starts[i+1]])
                    gc.collect() # gotta make sure we dont ever exceed 1TB

            # clear room for splits
            del kmer_matrix
            del kmer_cols
            gc.collect()

            # load, feature select, then save each split. force_per_class needs own rules: see below
            if not os.path.exists(os.path.abspath(os.path.curdir)+"data/multi-mer/kbest/{}/{}_{}_{}mer_matrix{}.npy".format(dataset, num_feats, drug, kmer_length, 1)) and force_per_class == 0:
                for i in range(10):
                    # feature selection
                    kmer_matrix = np.load("data/multi-mer/kbest/{}/all_{}_{}mer_matrix{}.npy".format(
                    dataset, drug, kmer_length, i+1))
                    kmer_cols = np.load("data/multi-mer/kbest/{}/all_{}_{}mer_cols{}.npy".format(
                    dataset, drug, kmer_length, i+1))

                    sk_obj = SelectKBest(f_classif, k = int(num_feats))
                    kmer_matrix = sk_obj.fit_transform(kmer_matrix, y_train)
                    kmer_cols = (sk_obj.transform(kmer_cols.reshape(1,-1))).flatten()

                    np.save("data/multi-mer/kbest/{}/{}_{}_{}mer_matrix{}.npy".format(
                    dataset, num_feats, drug, kmer_length, i+1), kmer_matrix)
                    np.save("data/multi-mer/kbest/{}/{}_{}_{}mer_cols{}.npy".format(
                    dataset, num_feats, drug, kmer_length, i+1), kmer_cols)

                    del kmer_matrix
                    del kmer_cols
                    gc.collect()

            # same block as directly above but for force_per_class
            if not os.path.exists(os.path.abspath(os.path.curdir)+"data/multi-mer/per_class/{}/{}_{}_{}mer_matrix1-1.npy".format(dataset, num_feats, drug, kmer_length)) and force_per_class == 1:
                for i in range(10):
                    # we are going to repeat the feature selection once per class that has >5 samples,
                    kmer_matrix = np.load("data/multi-mer/kbest/{}/all_{}_{}mer_matrix{}.npy".format(
                    dataset, drug, kmer_length, i+1))
                    kmer_cols = np.load("data/multi-mer/kbest/{}/all_{}_{}mer_cols{}.npy".format(
                    dataset, drug, kmer_length, i+1))

                    for class_num, mic_class in enumerate(mic_class_dict[drug]):
                        counts = Counter(y_train)
                        if counts[class_num] < 5:
                            print("skipping {} class {} because it only has {} samples".format(drug, mic_class, counts[mic_class]))
                            continue

                        class_y_train = [int(ele == class_num) for ele in y_train]

                        assert(np.sum(class_y_train) >=5)

                        sk_obj = SelectKBest(f_classif, k = int(num_feats))
                        split_matrix = sk_obj.fit_transform(kmer_matrix, class_y_train)
                        split_cols = (sk_obj.transform(kmer_cols.reshape(1,-1))).flatten()

                        np.save("data/multi-mer/per_class/{}/{}_{}_{}mer_matrix{}-{}.npy".format(
                        dataset, num_feats, drug, kmer_length, i+1, class_num+1), split_matrix)
                        np.save("data/multi-mer/per_class/{}/{}_{}_{}mer_cols{}-{}.npy".format(
                        dataset, num_feats, drug, kmer_length, i+1, class_num+1), split_cols)

                        del split_matrix
                        del split_cols
                        gc.collect()

            # merge results, do final feature selection
            kmer_matrix = []
            kmer_cols = []

            if(force_per_class == 0):
                for i in range(10):
                    if(i == 0):
                        kmer_matrix = np.load("data/multi-mer/kbest/{}/{}_{}_{}mer_matrix{}.npy".format(
                        dataset, num_feats, drug, kmer_length, i+1))
                    else:
                        split_matrix = np.load("data/multi-mer/kbest/{}/{}_{}_{}mer_matrix{}.npy".format(
                        dataset, num_feats, drug, kmer_length, i+1))
                        kmer_matrix = np.concatenate((kmer_matrix,split_matrix), axis=1)
                        del split_matrix
                    split_cols = np.load("data/multi-mer/kbest/{}/{}_{}_{}mer_cols{}.npy".format(
                    dataset, num_feats, drug, kmer_length, i+1))
                    for col in split_cols:
                        kmer_cols.append(col)

            else:
                for i in range(10):
                    for class_num, mic_class in enumerate(mic_class_dict[drug]):
                        counts = Counter(y_train)
                        print("Skipping class in append stage")
                        if counts[class_num] < 5:
                            continue
                        if (i == 0) and (class_num == 0):
                            kmer_matrix = np.load("data/multi-mer/per_class/{}/{}_{}_{}mer_matrix{}-{}.npy".format(
                            dataset, num_feats, drug, kmer_length, i+1, class_num+1))
                        else:
                            split_matrix = np.load("data/multi-mer/per_class/{}/{}_{}_{}mer_matrix{}-{}.npy".format(
                            dataset, num_feats, drug, kmer_length, i+1, class_num+1))
                            kmer_matrix = np.concatenate((kmer_matrix,split_matrix), axis=1)
                            del split_matrix
                        split_cols = np.load("data/multi-mer/per_class/{}/{}_{}_{}mer_cols{}-{}.npy".format(
                        dataset, num_feats, drug, kmer_length, i+1, class_num+1))
                        for col in split_cols:
                            kmer_cols.append(col)


            kmer_cols = np.array(kmer_cols)

            if(force_per_class == 0):

                sk_obj = SelectKBest(f_classif, k = int(num_feats))
                kmer_matrix = sk_obj.fit_transform(kmer_matrix, y_train)
                kmer_cols = (sk_obj.transform(kmer_cols.reshape(1,-1))).flatten()

                np.save("data/multi-mer/kbest/{}/{}_{}_{}mer_matrix.npy".format(
                dataset, num_feats, drug, kmer_length), kmer_matrix)
                np.save("data/multi-mer/kbest/{}/{}_{}_{}mer_cols.npy".format(
                dataset, num_feats, drug, kmer_length), kmer_cols)

            else:
                # can look into splitting and doing feature selection on each class of the matrix
                np.save("data/multi-mer/per_class/{}/{}_{}_{}mer_matrix.npy".format(
                dataset, num_feats, drug, kmer_length), kmer_matrix)
                np.save("data/multi-mer/per_class/{}/{}_{}_{}mer_cols.npy".format(
                dataset, num_feats, drug, kmer_length), kmer_cols)


    else:
        del kmer_matrix
        del kmer_cols
        gc.collect()

        kmer_matrix = np.load("data/multi-mer/kbest/{}/{}_{}_{}mer_matrix.npy".format(
        dataset, num_feats, drug, kmer_length))
        kmer_cols = np.load("data/multi-mer/kbest/{}/{}_{}_{}mer_cols.npy".format(
        dataset, num_feats, drug, kmer_length))

    print(len(kmer_cols))
    assert(len(kmer_cols) == int(num_feats) or force_per_class == 1)

    # calculate feature importances with anova to compare against model importances
    import scipy.stats as stats
    skb = SelectKBest(f_classif, k='all').fit(kmer_matrix, y_train)

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

    if(force_per_class == 0):
        feat_save = 'data/multi-mer/feat_ranks/{}_{}_{}_{}mer_feature_ranks.npy'.format(
        dataset,num_feats,drug,kmer_length)
    else:
        feat_save = 'data/multi-mer/per_class/{}_{}_{}_{}mer_feature_ranks.npy'.format(
        dataset,num_feats,drug,kmer_length)
    np.save(feat_save, np.vstack((kmer_cols, [float(i) for i in model.feature_importances_],[float(i) for i in skb.scores_])))
