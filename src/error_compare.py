"""
This script compares the errors of the 3 models to determine simularity,
requires error output from model.py and genome_error_table_converter.py
"""

import numpy as np
import pandas as pd
import os, sys

from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt

drug = 'AMP'

errors = {}
models = ['XGB','ANN','SVM']

def find_num_uniq(model_type):
    # find how many errors are unique to that model
    other_models = np.array(models)[[i != model_type for i in models]]
    uniqs = []
    for err in errors[model_type]:
        if err not in list(errors[other_models[0]]) or err not in list(errors[other_models[1]]):
            uniqs.append(err)
    return len(uniqs)

for model in models:

    df_folds = {}

    for fold in range(1,6):
        df_folds[fold] = pd.read_csv("data/errors/{}_1000feats_{}trainedOnpublic_testedOncv_fold{}.csv".format(drug, model, str(fold)))

    drug_df = pd.concat([df_folds[i] for i in range(1,6)])

    errors[model] = drug_df['Genome'][drug_df['OffByOne']==False]

print("XGB: {}, SVM: {}, ANN: {}".format(len(errors['XGB']),len(errors['SVM']),len(errors['ANN'])))
print("XGB,SVM: {}, XGB,ANN {}".format(len(list(set(errors['XGB']) & set(errors['SVM']))), len(list(set(errors['XGB']) & set(errors['ANN'])))))
print("SVM,ANN: {}".format(len(list(set(errors['SVM']) & set(errors['ANN'])))))
print("All: {}".format(len(list(set(errors['SVM']) & set(errors['ANN']) & set(errors['XGB'])))))

venn3([set(errors[model]) for model in models], set_labels = models, normalize_to = 10)

##for model in models:
#    print(list(errors[model])[0])
#sys.exit()
for model in models:
    print("unique to {}: {}".format(model,find_num_uniq(model)))

plt.show()
