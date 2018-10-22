
#################################################################

# Location of the MIC data
DRUGS = ["AMP","TIO"]
FEATURES_SIZES = [500]

#################################################################

import gc
import numpy as np
import pandas as pd
import pickle

from src.feature_selection import dropna_and_encode_rows

rule all:
    input:
        expand('data/{drug}/{feat}features/grdi_xgb_report.txt', drug=DRUGS,
            feat=FEATURES_SIZES),

rule encode_mics:
    input:
        "data/grdi_unfiltered/kmer_rows.npy",
        "data/grdi_mic_class_dataframe.pkl",
        "data/grdi_mic_class_order_dict.pkl"
    params:
        d="{drug}"
    output:
        "data/{drug}/y_grdi.pkl"
    run:
        rows = np.load(input[0])
        micdf = pd.read_pickle(input[1])
        orddict = pickle.load(open(input[2], 'rb'))

        y = dropna_and_encode_rows(micdf.loc[:,params.d], orddict[params.d])
        print(y.shape)
        y.to_pickle(output[0])

rule select:
    input:
        'data/grdi_unfiltered/kmer_matrix.npy',
        'data/grdi_unfiltered/kmer_cols.npy',
        'data/grdi_unfiltered/kmer_rows.npy',
        'data/{drug}/y_grdi.pkl',
        'data/{drug}/{feat}features/X_train.pkl'
    output:
        'data/{drug}/{feat}features/X_grdi.pkl',
        'data/{drug}/{feat}features/y_grdi_xref.pkl'
    run:

        kmer = np.load(input[0])
        cols = np.load(input[1])
        rows = np.load(input[2])
        y = pd.read_pickle(input[3])

        rows = [ r.decode("utf-8") for r in rows ]
        cols = [ c.decode("utf-8") for c in cols ]

        # Turn into DataFrame
        X_grdi = pd.DataFrame(kmer, index=rows, columns=cols)

        # Drop kmer rows with no MIC, drop MICs with no kmer row
        has_kmer = np.in1d(y.index, X_grdi.index)
        y = y[has_kmer]
        X_grdi = X_grdi.loc[y.index.values]
        print("Kmer matrix dim {}".format(X_grdi.shape))

        # Pull out relevant columns
        X_train = pd.read_pickle(input[4])
        columns = X_train.columns.values
        X_grdi = X_grdi.loc[:,columns]
        print("Filtered matrix dim {}".format(X_grdi.shape))
        print(X_grdi.head())

        y.to_pickle(output[1])
        X_grdi.to_pickle(output[0])


rule test:
    input:
        'data/{drug}/{feat}features/X_grdi.pkl',
        'data/{drug}/{feat}features/y_grdi_xref.pkl',
        'data/{drug}/{feat}features/xgb_model.pkl'
    params:
        d="{drug}"
    output:
        'data/{drug}/{feat}features/grdi_xgb_report.txt'
    run:
        X_grdi = pd.read_pickle(input[0])
        y_grdi = pd.read_pickle(input[1])

        print(X_grdi.head())
        print(y_grdi.head())

        model = pickle.load(open(input[2], 'rb'))
        y_pred = model.predict( X_grdi.values )
        acc = sum(y_grdi == y_pred)/len(y_grdi)
        print(acc)
        with open(output[0], 'w') as outfh:
            outfh.write("Xgboost accuracy for drug {}: {}\n".format(params.d, acc))
