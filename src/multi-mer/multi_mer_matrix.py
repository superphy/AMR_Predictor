"""
This program takes in a list of multi-mers and builds a matrix of them
"""

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import os, sys

if __name__ == "__main__":
    mm_path = sys.argv[1]
    split_num = int(sys.argv[2])
    kmer_length = sys.argv[3]

    if split_num in range(1,47):
        dataset = 'public'
    elif split_num in range(47,59):
        dataset = 'grdi'
    else:
        raise Exception("Split should be in range 1-59 but got ",split_num)

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
        relevant_feats = np.load(mm_path)

        cols_dict = { relevant_feats[i] : i for i in range(0, len(relevant_feats))}

        # Create a temp row to fill and return (later placed in the kmer_matrix)
        temp_row = [0]*len(relevant_feats)

        if(dataset == 'grdi'):
            jf_path = "data/grdi_genomes/jellyfish_results{}/{}".format(kmer_length, filename)
        else:
            jf_path = "data/genomes/jellyfish_results{}/{}".format(kmer_length, filename)
            
        for record in SeqIO.parse(jf_path, "fasta"):
            # Retrieve the sequence as a string
            kmer_seq = record.seq
            kmer_seq = kmer_seq._get_seq_str_and_check_alphabet(kmer_seq)

            """
            if(kmer_seq in relevant_feats):
                kmer_count = int(record.id)
                temp_row[cols_dict[kmer_seq]] = kmer_count
            else:
                raise Exception("kmer {} found in jellyfish file {} but not in master list {}".format(kmer_seq, filename,mm_path))
            """
            kmer_count = int(record.id)
            temp_row[cols_dict[kmer_seq]] = kmer_count

        return filename, temp_row

    mmers = np.load(mm_path)

    relevant_feats = mmers

    if(dataset == 'public'):
        genomes = np.load("data/multi-mer/genome_names.npy")
        start_num = (split_num - 1) * 128
        stop_num = split_num * 128
    elif(dataset == 'grdi'):
        genomes = np.load("data/multi-mer/grdi_genome_names.npy")
        start_num = (split_num-46 -1) * 128
        stop_num = (split_num-46) * 128

    if split_num in [46,58]:
        genomes = genomes[start_num:]
    else:
        genomes = genomes[start_num:stop_num]

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
    num_start += min(16,len(genomes))
    progress()
    with ProcessPoolExecutor(max_workers=16) as ppe:
        for genome_name,temp_row in ppe.map(make_row, genomes):
            num_stop+=1
            if(num_start<total):
                num_start+=1
            progress()
            for i, val in enumerate(temp_row):
                kmer_matrix[rows_dict[genome_name]][i] = val

    # save everything
    np.save("data/multi-mer/splits/{}mer_matrix{}.npy".format(kmer_length, split_num), kmer_matrix)
    np.save("data/multi-mer/splits/{}mer_rows{}.npy".format(kmer_length,split_num), runs)
    np.save("data/multi-mer/splits/{}mer_cols{}.npy".format(kmer_length,split_num), relevant_feats)
