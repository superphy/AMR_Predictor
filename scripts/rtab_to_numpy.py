#!/usr/bin/env python
import csv
import numpy as np
import pandas as pd
'''
Converts gene_presence_absence Rtab from roary to a npy matrix for use with machine learning models
'''
df = pd.read_csv("gene_presence_absence.Rtab", sep="\t")
genes = df['Gene']
np.save("gene_presence_absence_genes.npy", genes.values)
df_no_genes = df.drop('Gene', axis =1)
np.save("gene_presence_absence_matrix.npy", np.transpose(df_no_genes.values))
np.save("gene_presence_absence_genomes.npy", df_no_genes.columns.values)
