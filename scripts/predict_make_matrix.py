#!/usr/bin/env python

from Bio import SeqIO
from pathlib import Path
import numpy as np
import os
import sys
import itertools


"""
Creates a matrix of kmer counts, and two dictionaries to keep
track of row and column names & their indices.
"""

#####################################################################
## Input values
num_input_genomes = int(sys.argv[1])	# Set this to your number of input files.
kmer_size = int(sys.argv[2])			# Set this to your kmer length.
matrix_dtype = sys.argv[3]				# Set the data type for the matrix.
										#  ->Note uint8 has max kmercount of 256

input_fpath = sys.argv[4]
save_path = sys.argv[5]
#####################################################################

print("starting create matrix")

num_rows = num_input_genomes
num_cols = 4**kmer_size

# Initialize the matrix
kmer_matrix = np.zeros((num_rows,num_cols), dtype=matrix_dtype)

# Initialize the dictionaries of row and col names
col_names = {}
row_names = {}

# Create the column dictionary of sequences and their col index
chars = "AGCT"
i=0
for item in itertools.product(chars, repeat=kmer_size):
	col_names["".join(item)] = i
	i+=1

# Go through each results file
p = Path(input_fpath)
row_index = 0
for filename in p.iterdir():

	if row_index%100==0: print("row milestone: ",row_index)

	# Get the genomeid from the filepath
	genomeid = filename.name
	thefile = str(filename)
	genomeid = genomeid.split('.')[0]

	# Add the genome to the lsit of rows
	row_names[genomeid]= row_index

	temp_row = [0]*num_cols
	# Walk through the file
	for record in SeqIO.parse(thefile, "fasta"):
		# Retrieve the sequence as a sting
		kmerseq = record.seq
		kmerseq = kmerseq._get_seq_str_and_check_alphabet(kmerseq)	
			
		# Retrieve the kmer count as an int
		kmercount = record.id
		kmercount = int(kmercount)

		# Lookup the seq in the column list for the index
		col_index = col_names[kmerseq]

		# Put the kmercount in the right spot in the row
		temp_row[col_index] = kmercount

	# Put the row into the matrix
	kmer_matrix[row_index,:] = temp_row
	row_index+=1

print(kmer_matrix)
print("ending create matrix\n")

# Save the matrix and its dictionaries
if not os.path.exists('./'+save_path):
	os.mkdir(save_path)
np.save(save_path+'kmer_matrix.npy', kmer_matrix)
np.save(save_path+'dict_kmer_rows.npy', row_names)
np.save(save_path+'dict_kmer_cols.npy', col_names)