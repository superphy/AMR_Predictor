#!/usr/bin/env python
import csv
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import os

###########################################################################################
# Load, convert, and save the roary output

df = pd.read_csv("roary_out/gene_presence_absence.Rtab", sep="\t")
genes = df['Gene']
np.save("roary_out/gene_presence_absence_genes.npy", genes.values)
df_no_genes = df.drop('Gene', axis =1)
np.save("roary_out/gene_presence_absence_matrix.npy", np.transpose(df_no_genes.values))
np.save("roary_out/gene_presence_absence_genomes.npy", df_no_genes.columns.values)



###########################################################################################
# Prep the data per drug for predictions

# Matrix of experimental MIC values
df = joblib.load("amr_data/mic_class_dataframe.pkl")
df_rows = df.index.values 	# Row names are genomes
df_cols = df.columns		# Col names are drugs

# Matrix of classes for each drug
mic_class_dict = joblib.load("amr_data/mic_class_order_dict.pkl")

# Load the filtered kmer matrix and its row and col lookups
matrix = np.load("roary_out/gene_presence_absence_matrix.npy")
mat_cols = np.load("roary_out/gene_presence_absence_genes.npy")
mat_rows = np.load("roary_out/gene_presence_absence_genomes.npy")

if not os.path.exists('./roary_amr/'):
	os.mkdir('roary_amr/')

# For each drug
for drug in df_cols:
	print("start: prepping amr data for ",drug)

	# Create a directory for the drugs' data
	if not os.path.exists('./roary_amr/'+drug):
		os.mkdir('roary_amr/'+drug)

	# Get the column index of the drug
	col_index = df.columns.get_loc(drug)
	# Set invalid entries to be NaN for easy deletion
	for index, row in df.iterrows():
		x=row[col_index]
		if str(x) == 'invalid':
			row[col_index] = np.NaN
	# Drop all rows where the cell has an NaN entry
	new_df = df.dropna(subset=[drug])
	new_df = new_df[drug]
	new_df_rows = new_df.index.values 	# Row names are genomes

	# Mask the matrix
	num_rows = len(df_rows)
	mask = [1]*(num_rows)

	for i in range(mat_rows.shape[0]):
		x = mat_rows[i]#.decode('utf-8')
		#print(x)
		#print(new_df_rows)
		if x not in new_df_rows:
			#print("ahhhhhhhhhhhhh")
			mask[i] = 0
		#else: print("ahhh")
	bool_mask = [bool(x) for x in mask]

	print(matrix.shape)
	new_matrix = matrix[bool_mask, :]
	new_rows   = mat_rows[bool_mask]
	print(new_matrix.shape)

	# Save the row names (genomes) so that we dont have
	# to make a copy of it to manipulate it (time saver)
	np.save('roary_amr/'+drug+'/matrix_rows_genomes.npy', new_rows)
	# Lookup the MIC value for each genome and replace the genome
	# name with that value, and save it as a separate np array
	for i in range(new_rows.shape[0]):
		gen = new_rows[i]#.decode('utf-8')
		row_index = np.where(new_df_rows==gen)
		mic_val = new_df.iloc[row_index][0]
		new_rows[i] = mic_val

	joblib.dump(new_df, 'roary_amr/'+drug+'/mic_df.pkl')
	np.save('roary_amr/'+drug+'/matrix.npy', new_matrix)
	np.save('roary_amr/'+drug+'/matrix_rows_mic.npy', new_rows)
	np.save('roary_amr/'+drug+'/matrix_cols.npy', mat_cols)

	print("end: prepping amr data for ",drug)


