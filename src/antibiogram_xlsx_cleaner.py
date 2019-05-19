#!/usr/bin/env python

"""
This program loads in the antibiogram excel sheet and cleans it,
preparing for the MIC classifications of mics.snakefile
"""

import numpy as np
import pandas as pd
import os, sys

def remove_tail(seq):
    # Will turn "<= 1 mg/L" or "<= 1/0.5 mg/L" into "<= 1"
    if str(seq) == 'nan':
        return seq
    splits = seq.split('/')
    if len(splits)==2:
        return splits[0][:-3]
    elif len(splits)==3:
        return splits[0]
    else:
        raise Exception("Unexpected number of / seen")

def remove_equality(seq):
    # Will turn "== 2" into "2" || '== 2 mg/L' into '2 mg/L'
    if str(seq) == 'nan':
        return seq
    if seq[0:2] == '==':
        return seq[3:]
    else:
        return seq

mic_df = pd.read_excel('data/2019-Antibiogram.xlsx')

mics = ["MIC_AMP", "MIC_AMC", "MIC_FOX", "MIC_CRO",
"MIC_TIO", "MIC_GEN", "MIC_FIS", "MIC_SXT", "MIC_AZM",
"MIC_CHL", "MIC_CIP", "MIC_NAL", "MIC_TET"]

# go through every column of relevant MIC values
for mic in mics:
    mic_df[mic]=[remove_equality(i) for i in [remove_tail(i) for i in mic_df[mic]]]

mic_df.to_excel("data/new_antibiogram_test.xlsx")
