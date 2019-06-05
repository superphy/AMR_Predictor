"""
This program takes in a list of 15mers and builds a matrix of 15 mers
"""

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import os, sys

if __name__ == "__main__":
    OxF_mers_path = sys.argv[1]
    drug = sys.argv[2]

    num_start = 0
    num_stop = 0
    total = 0

    def progress():
        sys.stdout.write('\r')
        sys.stdout.write("Loading Genomes: {} started, {} finished, {} total".format(num_start,num_stop,total))
        #sys.stdout.flush()
        if(num_stop==total):
            print("\nAll Genomes Loaded!\n")

    def make_row(filename):
        from Bio import Seq, SeqIO
        import numpy as np
        """
        Given a genome file, create and return a row of kmer counts
        to be inserted into the kmer matrix.
        """
        # flatten 2d list
        relevant_feats = []
        for row in np.load(OxF_mers_path):
            for ele in row:
                relevant_feats.append(ele)

        cols_dict = { relevant_feats[i] : i for i in range(0, len(relevant_feats))}

        # Create a temp row to fill and return (later placed in the kmer_matrix)
        temp_row = [0]*len(relevant_feats)

        # Walk through the file
        for record in SeqIO.parse("data/jellyfish_results15/"+filename, "fasta"):
            # Retrieve the sequence as a string
            kmer_seq = record.seq
            kmer_seq = kmer_seq._get_seq_str_and_check_alphabet(kmer_seq)


            if(kmer_seq in relevant_feats):
                kmer_count = int(record.id)
                temp_row[cols_dict[kmer_seq]] = kmer_count

        return filename, temp_row

    # flatten 2d list
    OxF_mers = []
    for row in np.load(OxF_mers_path):
        for ele in row:
            OxF_mers.append(ele)

    relevant_feats = OxF_mers

    genomes = ([files for r,d,files in os.walk("data/jellyfish_results15/")][0])
    total = len(genomes)
    runs = [i.split('.')[0] for i in genomes]

    # declaring empty kmer matrix to fill
    kmer_matrix = np.zeros((len(genomes),len(relevant_feats)),dtype = 'uint8')

    # making dicts for faster indexing
    # note that rows dict is in filenames not genome/run names
    rows_dict = { genomes[i] : i for i in range(0, len(genomes))}
    cols_dict = { relevant_feats[i] : i for i in range(0, len(relevant_feats))}

    # Use concurrent futures to get multiple rows at the same time
    # Then place completed rows into the matrix and update the row dictionary
    num_start += min(48,len(genomes))
    progress()
    with ProcessPoolExecutor(max_workers=48) as ppe:
        for genome_name,temp_row in ppe.map(make_row, genomes):
            num_stop+=1
            if(num_start<total):
                num_start+=1
            progress()
            for i, val in enumerate(temp_row):
                kmer_matrix[rows_dict[genome_name]][i] = val

    # save everything
    np.save("annotation/15mer_data/{}_kmer_matrix.npy".format(drug), kmer_matrix)
    np.save("annotation/15mer_data/{}_kmer_rows.npy".format(drug), runs)
    np.save("annotation/15mer_data/{}_kmer_cols.npy".format(drug), relevant_feats)
