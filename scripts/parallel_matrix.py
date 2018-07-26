#!/usr/bin/env python

from Bio import SeqIO
from pathlib import Path
import numpy as np
import os
import sys
import itertools
import re
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

def get_files_to_analyze(file_or_directory):
    """
    :param file_or_directory: name of either file or directory of files
    :return: a list of all files (even if there is only one)
    """

    # This will be the list we send back
    files_list = []

    if os.path.isdir(file_or_directory):
        # Using a loop, we will go through every file in every directory
        # within the specified directory and create a list of the file names
        for root, dirs, files in os.walk(file_or_directory):
            for filename in files:
                files_list.append(os.path.join(root, filename))
    else:
        # We will just use the file given by the user
        files_list.append(os.path.abspath(file_or_directory))

    # We will sort the list when we send it back, so it always returns the
    # same order
    return sorted(files_list)


def make_row(filename):
    """
    Given a genome file, create and return a row of kmer counts
    to be inerted into the mer matrix.
    """
    # Filepath
    thefile = str(filename[0])

    # Get the genome id from the filepath
    genomeid = filename[0].split('/')[-1]
    genomeid = genomeid.split('.')[-2]

    # Create a temp row to fill and return (later placed in the kmer_matrix)
    temp_row = [0]*num_cols

    # Walk through the file
    for record in SeqIO.parse(thefile, "fasta"):
        # Retrieve the sequence as a string
        kmerseq = record.seq
        kmerseq = kmerseq._get_seq_str_and_check_alphabet(kmerseq)  
                
        # Retrieve the kmer count as an int
        kmercount = record.id
        kmercount = int(kmercount)

        # Lookup the seq in the column list for the index
        col_index = col_names[kmerseq]

        # Put the kmercount in the right spot in the row
        temp_row[col_index] = kmercount

    return genomeid,temp_row


if __name__ == "__main__":
    """
    Creates a matrix of kmer counts, and two dictionaries to keep
    track of row and column names & their indices.
    """
    print("start: create matrix (parallel)")

    #####################################################################
    ## Input values
    num_input_genomes = int(sys.argv[1])    # Set this to your number of input files.
    kmer_size = int(sys.argv[2])            # Set this to your kmer length.
    matrix_dtype = sys.argv[3]              # Set the data type for the matrix.
                                            #  ->Note uint8 has max kmercount of 256  
    results_path = str(sys.argv[4])                             
    #####################################################################
    
    # Initialize the kmer matrix
    num_rows    = num_input_genomes
    num_cols    = 4**kmer_size
    kmer_matrix = np.zeros((num_rows,num_cols), dtype=matrix_dtype)

    # Initialize the dictionaries of row and col names
    col_names = {}
    row_names = {}

    # Create the column dictionary of sequences and their col index
    chars = "AGCT"
    i = 0
    for item in itertools.product(chars, repeat=kmer_size):
        col_names["".join(item)] = i
        i += 1

    # Get a list of all files and reshape it to use with concurrent futures
    files = get_files_to_analyze(results_path)
    x = np.asarray(files)
    y = x.reshape((len(x),1))

    # Use concurent futures to get multiple rows at the same time
    # Then place completed rows into the matrix and update the row dictionary
    row_index = 0
    with ProcessPoolExecutor(max_workers=cpu_count()) as ppe:
        for genomeid,temp_row in ppe.map(make_row, y):
            row_names[genomeid] = row_index
            kmer_matrix[row_index,:] = temp_row
            row_index += 1

    # Save the matrix and its dictionaries
    if not os.path.exists('./unfiltered'):
        os.mkdir('unfiltered')
    np.save('unfiltered/kmer_matrix.npy', kmer_matrix)
    np.save('unfiltered/dict_kmer_rows.npy', row_names)
    np.save('unfiltered/dict_kmer_cols.npy', col_names)

    print("end: create matrix (parallel)")




