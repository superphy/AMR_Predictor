"""
takes a 15mer and counts prevalence per class
"""

import numpy as np
import os, sys
from sklearn.externals import joblib

drug = 'AZM'
OxF_mer = 'CGGATGCAGTAAAAC'

kmer_matrix = np.load("annotation/15mer_data/AZM_kmer_matrix.npy")
kmer_cols = np.load("annotation/15mer_data/AZM_kmer_cols.npy")
OxF_mer_index = list(kmer_cols).index(OxF_mer)
kmer_rows = np.load("annotation/15mer_data/AZM_kmer_rows.npy")

mic_rows = [i.decode("utf-8") for i in np.load("data/AZM/kmer_rows_mic.npy")]
genomes_rows = [i.decode("utf-8") for i in np.load("data/AZM/kmer_rows_genomes.npy")]

# basically kmer rows mic
mic_15mer = []


for genome in kmer_rows:
    try:
        genome_index = genomes_rows.index(genome)
        mic_15mer.append(mic_rows[genome_index])
    except:
        mic_15mer.append('0')

mic_class_dict = joblib.load("predict/genomes/public_mic_class_order_dict.pkl")

for mic_class in mic_class_dict['AZM']:
    num_samples = 0
    times_seen = 0
    times_over_one = 0
    for i, mic in enumerate(mic_15mer):
        if mic == mic_class:
            count = kmer_matrix[i][OxF_mer_index]
            num_samples +=1
            times_seen += count
            if(count > 0):
                times_over_one += 1

    print("{} was seen in {} sequences an average of {} times (seen in {} of {} sequences)".format(OxF_mer, mic_class, times_seen/num_samples, times_over_one, num_samples))
