#!/usr/bin/env python

#import modules
import pandas as pd
import numpy as np
import os
import sys
keep_serovars = ['Kentucky','Heidelberg']
SISTR_path = "data/no_ecoli_GenotypicAMR_Master.xlsx"
NCBI_path = "data/dec2019_master_antibiogram.csv"

def which_serovar(NCBI, SISTR):
    if NCBI in [' ', 'Unknown','Serotype pending']:
        return SISTR
    else:
        return NCBI

if __name__ == "__main__":
    #load arrays from spreadsheet
    SISTR_df = pd.read_excel(SISTR_path)
    NCBI_df = pd.read_csv(NCBI_path)

    drug = sys.argv[1]

    #load genomes from kmer rows
    kmer_rows = np.load(os.path.abspath(os.path.curdir)+'/data/'+drug+"/kmer_rows_genomes.npy")

    #load kmer matrix for bool mask later
    kmer_matrix = np.load(os.path.abspath(os.path.curdir)+'/data/'+drug+"/kmer_matrix.npy")

    #load kmer mic rows
    kmer_mic = np.load(os.path.abspath(os.path.curdir)+'/data/'+drug+"/kmer_rows_mic.npy")

    runs = [i.decode('utf-8') for i in kmer_rows]

    possible_serovars = []

    for run in runs:
        possible_serovars.append([NCBI_df[NCBI_df['run']==run]['serovar'].values[0],SISTR_df[SISTR_df['run']==run]['serovar'].values[0]])

    serovars = [which_serovar(i[0],i[1]) for i in possible_serovars]

    bool_mask = [i in keep_serovars for i in serovars]

    #filter rows and matrix to contain specified serovar(s)
    kmer_rows = kmer_rows[bool_mask]
    kmer_matrix = kmer_matrix[bool_mask]
    kmer_mic = kmer_mic[bool_mask]

    if not os.path.exists(os.path.abspath(os.path.curdir)+'/data/kh_' + drug):
        os.mkdir(os.path.abspath(os.path.curdir)+'/data/kh_' + drug)

    #save the new array and matrix for that serovar
    np.save(os.path.abspath(os.path.curdir)+'/data/kh_'+drug+'/kmer_rows_genomes.npy', kmer_rows)
    np.save(os.path.abspath(os.path.curdir)+'/data/kh_'+drug+'/kmer_matrix.npy', kmer_matrix)
    np.save(os.path.abspath(os.path.curdir)+'/data/kh_'+drug+'/kmer_rows_mic.npy', kmer_mic)
