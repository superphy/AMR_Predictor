
#################################################################

# Location of the MIC data
DRUGS = ["AMP","TIO"]
FEATURES_SIZES = [500]

#################################################################

import pandas as pd
import numpy as np
import logging
import pickle

from scripts.feature_selection import select, dropna_and_encode_rows
from sklearn.model_selection import train_test_split

rule all:
    input:
        expand('amr_data/{drug}/{feat}features/X_train.pkl', drug=DRUGS,
            feat=FEATURES_SIZES),
        expand('amr_data/{drug}/{feat}features/X_test.pkl', drug=DRUGS,
            feat=FEATURES_SIZES),
        expand('amr_data/{drug}/{feat}features/y_train.pkl', drug=DRUGS,
            feat=FEATURES_SIZES),
        expand('amr_data/{drug}/{feat}features/y_test.pkl', drug=DRUGS,
            feat=FEATURES_SIZES)
        #expand('amr_data/{drug}/y.pkl', drug=DRUGS, feat=FEATURES_SIZES)

rule encode_mics:
    input:
        "amr_data/public_mic_class_dataframe.pkl",
        "amr_data/public_mic_class_order_dict.pkl"
    params:
        d="{drug}"
    output:
        "amr_data/{drug}/y.pkl"
    run:
        micdf = pd.read_pickle(input[0])
        orddict = pickle.load(open(input[1], 'rb'))

        y = dropna_and_encode_rows(micdf.loc[:,params.d], orddict[params.d])
        print(y.shape)
        y.to_pickle(output[0])

rule select:
    input:
        'unfiltered/kmer_matrix.npy',
        'unfiltered/kmer_cols.npy',
        'unfiltered/kmer_rows.npy',
        'amr_data/{drug}/y.pkl'
    params:
        fmax=0.99,
        fmin=0.01,
        n="{feat}"
    output:
        'amr_data/{drug}/{feat}features/X_train.pkl',
        'amr_data/{drug}/{feat}features/X_test.pkl',
        'amr_data/{drug}/{feat}features/y_train.pkl',
        'amr_data/{drug}/{feat}features/y_test.pkl'
    run:

        kmer = np.load(input[0])
        cols = np.load(input[1])
        rows = np.load(input[2])
        y = pd.read_pickle(input[3])

        rows = [ r.decode("utf-8") for r in rows ]
        cols = [ c.decode("utf-8") for c in cols ]

        kdf = pd.DataFrame(kmer, index=rows, columns=cols)
        kdf = kdf.loc[y.index]
        print("Kmer matrix dim {}".format(kdf.shape))

        # Select train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            kdf, y, test_size=0.2, random_state=22)

        # Since these are pandas objects, indices are included
        print("X training dim: {}".format(X_train.shape))
        print("X test dim: {}".format(X_test.shape))
        print("y training dim: {}".format(y_train.shape))
        print("y test dim: {}".format(y_test.shape))

        idx = select(X_train, y_train, float(params.fmax), float(params.fmin), int(params.n))

        X_train = X_train.iloc[:,idx]
        X_test = X_test.iloc[:,idx]

        print("X training dim after selection: {}".format(X_train.shape))
        print("X test dim after selection: {}".format(X_test.shape))

        X_train.to_pickle(output[0])
        X_test.to_pickle(output[1])
        y_train.to_pickle(output[2])
        y_test.to_pickle(output[3])
