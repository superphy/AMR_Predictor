#!/usr/bin/env python

import numpy as np
from math import floor
import os

print("start: filter matrix")

# Loaad in the kmermatrix and row and col lookups
matrix = np.load("unfiltered/kmer_matrix.npy")
kmer_rows = np.load("unfiltered/kmer_rows.npy")
kmer_cols = np.load("unfiltered/kmer_cols.npy")

# Get the dimensions of the matrix
num_rows = np.shape(matrix)[0]
num_cols = np.shape(matrix)[1]

# Define cutoff values - when we should delete a column
low_cutoff = floor(num_rows*0.01)
high_cutoff = floor(num_rows*0.99)

# Create a mask. 1's are locations we want to keep.
# We set indices to 0 for what we want to delete.
delete_this = [1]*num_cols

# Walk through the columns
col_index = 0
while col_index < num_cols:
	if col_index%100000 == 0: print("col milestone: ", col_index)
	# Count the number of empty spots in the column
	empty_count = np.bincount(matrix[:,col_index])[0]
	# Calculate the number of filled spots in the column
	fill_count = num_rows - empty_count
	# If the cound is beyond our thresholds, mark the index to be deleted
	if fill_count <= low_cutoff or fill_count >= high_cutoff:
		delete_this[col_index] = 0
	col_index+=1

delete_this = [bool(x) for x in delete_this]
delete_this = np.array(delete_this)

print(matrix.shape)
# Apply the mask to the matrix to delete unwanted rows
matrix  = matrix[:, delete_this]
print(matrix.shape)

# Apply the mask to the column lookup to sdelete unwanted rows
kmer_cols = kmer_cols[delete_this]

# Save the filtered matrix and its row and column lookups
if not os.path.exists('./filtered'):
	os.mkdir('filtered')
np.save("filtered/filtered_matrix.npy", matrix)
np.save("filtered/filtered_cols.npy", kmer_cols)
np.save("filtered/filtered_rows.npy", kmer_rows)

print("end: filter matrix")