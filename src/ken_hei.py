#!/usr/bin/env python

#import modules
import pandas as pd
import numpy as np
import os
import sys
SEROVAR1 = "Heidelberg"
SEROVAR2 = "Kentucky"
FILEPATH = "data/no_ecoli_GenotypicAMR_Master.xlsx"

if __name__ == "__main__":
    #load arrays from spreadsheet
    df = pd.read_excel(FILEPATH)
    df_run = df['run'].tolist()
    df_serovar = df['serovar'].tolist()
    drug = sys.argv[1]

    #load genomes from kmer rows
    kmer_rows = np.load(os.path.abspath(os.path.curdir)+'/data/'+drug+"/kmer_rows_genomes.npy")

    #load kmer matrix for bool mask later
    kmer_matrix = np.load(os.path.abspath(os.path.curdir)+'/data/'+drug+"/kmer_matrix.npy")

    #load kmer mic rows
    kmer_mic = np.load(os.path.abspath(os.path.curdir)+'/data/'+drug+"/kmer_rows_mic.npy")

    #funtion to find index of a genome in the run array
    def find_index(genome):
        for i, element in enumerate(kmer_rows):
            if genome == element:
                return i

    bool_mask = np.zeros(len(kmer_rows))

    #funtion to fill the bool mask by comparing run genomes against kmer rows genomes based on their serovar
    def find_serovar(kmer_rows):
        for i, element in enumerate(kmer_rows):
            if df_serovar[find_index(element)] == SEROVAR1 or df_serovar[find_index(element)] == SEROVAR2:
                bool_mask[i] = 1

    find_serovar(kmer_rows)
    bool_mask = [i == 1 for i in bool_mask]

    #filter rows and matrix to contain specified serovar(s)
    kmer_rows = kmer_rows[bool_mask]
    kmer_matrix = kmer_matrix[bool_mask, :]
    kmer_mic = kmer_mic[bool_mask]

    if not os.path.exists(os.path.abspath(os.path.curdir)+'/data/kh_' + drug):
        os.mkdir(os.path.abspath(os.path.curdir)+'/data/kh_' + drug)

    #save the new array and matrix for that serovar
    np.save(os.path.abspath(os.path.curdir)+'/data/kh_'+drug+'/kmer_rows.npy', kmer_rows)
    np.save(os.path.abspath(os.path.curdir)+'/data/kh_'+drug+'/kmer_matrix.npy', kmer_matrix)
    np.save(os.path.abspath(os.path.curdir)+'/data/kh_'+drug+'/kmer_rows_mic.npy', kmer_mic)
