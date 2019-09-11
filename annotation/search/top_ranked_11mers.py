"""
This script merges top features scores into a single readable csv
"""
import pandas as pd
import numpy as np
import os, sys
row = []

drugs=["AMP","AMC","AZM","CHL","CIP","CRO","FIS","FOX","GEN","NAL","SXT","TET","TIO"]
datasets=["public","grdi"]

for drug in drugs:
    for dataset in datasets:
        if dataset == 'grdi' and drug in ['AZM','FIS']:
            continue
        imp_arr = np.load("annotation/search/features_scores/{}_{}.npy".format(drug,dataset))
        for kmer, score in imp_arr:
            row.append([dataset,drug,kmer,score])

df = pd.DataFrame(data=row, columns = ['Dataset','Antimicrobial','11mer','Importance'])

print(df)

df.to_csv("annotation/search/features_scores/score_summary.csv")
