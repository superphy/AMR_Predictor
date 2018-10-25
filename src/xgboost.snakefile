
#################################################################

# Location of the MIC data
DRUGS = ["AMP","TIO"]
FEATURES_SIZES = [500]

#################################################################

import pandas as pd
import pickle
from hpsklearn import HyperoptEstimator, xgboost_classification
from hyperopt import tpe

from model_evaluators import xgb_tester


rule all:
    input:
        expand('data/{drug}/{feat}features/xgb_model.pkl', drug=DRUGS,
            feat=FEATURES_SIZES),
        expand('data/{drug}/{feat}features/xgb_report.txt', drug=DRUGS,
            feat=FEATURES_SIZES)

rule train:
    input:
        'data/{drug}/{feat}features/X_train.pkl',
        'data/{drug}/{feat}features/X_test.pkl',
        'data/{drug}/{feat}features/y_train.pkl',
        'data/{drug}/{feat}features/y_test.pkl'
    params:
        d="{drug}"
    output:
        'data/{drug}/{feat}features/xgb_model.pkl',
        'data/{drug}/{feat}features/xgb_report.txt'
    run:
        X_train = pd.read_pickle(input[0])
        X_test = pd.read_pickle(input[1])
        y_train = pd.read_pickle(input[2])
        y_test = pd.read_pickle(input[3])

        print(X_train.head())
        print(y_train.head())

        # Due to bug in Hyperopt-Sklearn, indexed/named matrices produce error
        param_search = HyperoptEstimator( classifier=xgboost_classification('xbc'), preprocessing=[], algo=tpe.suggest, trial_timeout=2000 )
    	param_search.fit( X_train.values, y_train.values )
    	model = param_search.best_model()['learner']

        pickle.dump(model, open(output[0], 'wb'))
        y_pred = model.predict( X_test.values )
        acc = sum(y_test == y_pred)/len(y_test)

        acc_0d, mcc, pred, act = xgb_tester(model, X_test.values, y_test, 0)
        acc_1d= xgb_tester(model, X_test.values, y_test, 1)[0]
        print(acc)
        print("*************************************************")
        print("Direct: {} 1D: {} MCC: {}".format(acc_0d, acc_1d, mcc))

        with open(output[1], 'w') as outfh:
            outfh.write("Xgboost accuracy for drug {}: {}\n".format(params.d, acc))
