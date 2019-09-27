"""
This script checks that the values in the matrix were generated correctly
Grabs 144 random genomes
"""

import numpy as np
import os, sys
import random
import time
from concurrent.futures import ProcessPoolExecutor

def check_row(data_row, genome_path, cols_dict):
    for record in SeqIO.parse(genome_path, "fasta"):
        # Retrieve the sequence as a string
        kmer_seq = record.seq
        kmer_seq = kmer_seq._get_seq_str_and_check_alphabet(kmer_seq)
        kmer_count = int(record.id)

        if kmer_count != data_row[cols_dict[kmer_seq]]:
            return "count of {} found in genome at {} for kmer {} but {} in matrix".format(kmer_count, genome_path, kmer_seq, data_row[cols_dict[kmer_seq]])

    return 1


def get_path(kmer_length, dataset, filename):
    if(dataset == 'grdi'):
        jf_path = "data/grdi_genomes/jellyfish_results{}/{}".format(kmer_length, filename)
    else:
        jf_path = "data/genomes/jellyfish_results{}/{}".format(kmer_length, filename)
    return jf_path+'.fa'

if __name__  == "__main__":
    dataset = sys.argv[1]
    kmer_length = sys.argv[2]

    if dataset == 'public':
        d_path = ''
    else:
        d_path = 'grdi_'


    kmer_matrix = np.load("data/multi-mer/{}{}mer_matrix.npy".format(d_path, kmer_length))
    kmer_rows = np.load("data/multi-mer/{}{}mer_rows.npy".format(d_path, kmer_length))
    kmer_cols = np.load("data/multi-mer/{}{}mer_cols.npy".format(d_path, kmer_length))

    genomes = ([files for r,d,files in os.walk("data/{}genomes/jellyfish_results{}/".format(d_path,kmer_length))][0])

    cols_dict = { kmer_cols[i] : i for i in range(0, len(kmer_cols))}
    rows_dict = { kmer_rows[i] : i for i in range(0, len(kmer_rows))}

    # grab 144 random genomes
    random.seed(time.time())
    random.shuffle(genomes)
    genomes = genomes[:144]

    keep_rows = [i.split('.')[0] for i in genomes]

    # only keep the information about the 144 randomly selected genomes
    kmer_matrix = kmer_matrix[[i in keep_rows for i in kmer_rows]]
    kmer_rows = kmer_rows[[i in keep_rows for i in kmer_rows]]
    repeat_cols = [cols_dict for i in range(144)]

    file_paths = [get_path(kmer_length, dataset, i) for i in kmer_rows]

    with ProcessPoolExecutor(max_workers=18) as ppe:
        for eval in ppe.map(check_row, zip(kmer_matrix, file_paths,repeat_cols)):
            if eva != 1:
                print(eval)

    print('Matrix Check Complete')
