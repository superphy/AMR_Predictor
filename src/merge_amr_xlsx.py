"""
This program merges old data with new 2019 amr data
"""

import numpy as np
import pandas as pd
import os,sys

def find_index(arr, element):
    for i, val in enumerate(arr):
        if(val == element):
            return i

cols = ['run', 'biosample',"MIC_AMP", "MIC_AMC", "MIC_FOX", "MIC_CRO","MIC_TIO", "MIC_GEN", "MIC_FIS",
"MIC_SXT", "MIC_AZM", "MIC_CHL", "MIC_CIP", "MIC_NAL", "MIC_TET",'source',
'cgmlst_matching_alleles', 'cgmlst_subspecies', 'h1', 'h2', 'o_antigen', 'serogroup',
'serovar', 'serovar_antigen', 'serovar_cgmlst']

new_df = pd.read_excel('new_antibiogram_test.xlsx')

old_df = pd.read_excel('no_ecoli_GenotypicAMR_Master1.xlsx')

mic_df = pd.DataFrame()


for col in cols:
    new_col = np.hstack((np.asarray(old_df[col]),np.asarray(new_df[col])))
    mic_df[col]= new_col

all_runs = list(set(mic_df['run']))

run_mask = np.zeros((len(mic_df['run'])))

# We only want 5853 runs

for i, run in enumerate(mic_df['run']):
    if run in all_runs:
        all_runs[find_index(all_runs,run)] = 'deleted'
        run_mask[i] = 1
run_mask = [i==1 for i in run_mask]
print(sum(run_mask))


mic_df[run_mask].to_excel("no_ecoli_GenotypicAMR_Master.xlsx")
