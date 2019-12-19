"""
Takes in a new and old dataframe and updates the old one with any
new information that might be available.
Works on the assumptions that any MICs currently matching between the NCBI
and PATRIC are correct
"""

import numpy as np
import pandas as pd
import os, sys
from collections import Counter

mics = ['AMC','AMP','AZM','FOX','TIO','CRO','CHL','CIP','GEN','NAL','FIS',
'TET','SXT']

def merge_dataset(old_df, new_df):
    """
    If a run exists in the new df, take that, else take the data in the old df
    """
    old_df = old_df.drop(columns = 'Unnamed: 0')
    new_df = new_df.drop(columns = ['Unnamed: 0', 'Unnamed: 0.1'])

    new_df.rename(columns={'BioSample': 'biosample'}, inplace=True)

    old_runs = old_df['run']
    new_runs = new_df['run']

    # remove any rows from the old df if that same data is available in the new dataset
    stripped_old = old_df[~old_df['run'].isin(new_runs)]

    # now merge stripped old with the new df
    common_cols = [i for i in old_df.columns.values if i in new_df.columns.values]

    return pd.merge(new_df, stripped_old, on=common_cols, how='outer')

def remove_conflicting(old_df, diff_df):
    """
    Takes in the old df and removes any MICs found in conflict
    """
    diffs = zip(list(diff_df['id']),list(diff_df['antimicrobial']))

    # for every difference, set that cell to blank
    for run, mic in diffs:
        old_df.loc[old_df['run']==run, 'MIC_'+mic] = ''

    return old_df

if __name__ == '__main__':

    # load old and update datasheets
    old_master_df = pd.read_excel("data/no_ecoli_GenotypicAMR_Master.xlsx")
    new_df = pd.read_csv("data/dec_2019_antibiogram_clean.csv")

    new_master_df = merge_dataset(old_master_df, new_df)

    # check to make sure no 2 biosamples are enter twice, one with unlabeled run
    duplicate_biosamples = ([i for i, count in Counter(list(new_master_df['biosample'])).items() if count >1])
    assert(len(duplicate_biosamples) == 0)

    # load in the differences found from compare_mic_df() in predict/PATRIC3.smk
    diff_df = pd.read_csv("PATRIC_and_NCBI_differences.csv")

    # remove any MICs that are currently in conflict between databases
    new_master_df = remove_conflicting(new_master_df, diff_df)

    new_master_df.to_csv('data/dec2019_master_antibiogram.csv')

'''
def std_mic(mic):
    return str(mic).lstrip('<>=').rstrip('.0')

def to_update(old_df, new_df):
    """
    Takes in 2 raw datasheets, returns a df of true difference
    """

    intersect = set(new_df['run']).intersection(set(old_df['run']))

    update_list = []

    for run in intersect:
        for drug in mics:
            old_val = old_df.loc[old_df['run']==run, 'MIC_'+drug].iloc[0]
            new_val = new_df.loc[new_df['run']==run, 'MIC_'+drug].iloc[0]
            if std_mic(old_val) != std_mic(new_val):
                update_list.append([run, drug, new_val])
                print(old_val, new_val)

    return update_list
'''
