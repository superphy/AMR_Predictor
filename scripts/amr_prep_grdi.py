#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.externals import joblib
import os

if __name__ == "__main__":
	# Matrix of experimental MIC values
	df = joblib.load(os.path.abspath(os.path.curdir)+"/amr_data/mic_class_dataframe.pkl")
	df_rows = df.index.values 	# Row names are genomes
	df_cols = df.columns		# Col names are drugs

	# Matrix of classes for each drug
	mic_class_dict = joblib.load(os.path.abspath(os.path.curdir)+"/amr_data/mic_class_order_dict.pkl")

	# Load the filtered kmer matrix and its row and col lookups
	kmer_matrix = np.load(os.path.abspath(os.path.curdir)+"/filtered/filtered_matrix.npy")
	kmer_cols = np.load(os.path.abspath(os.path.curdir)+"/filtered/filtered_cols.npy")
	kmer_rows = np.load(os.path.abspath(os.path.curdir)+"/filtered/filtered_rows.npy")


	############################################
	# need to filter out genomes
	# if df_genome not in kmer_rows, then delete the row from the df

	kmer_rows=kmer_rows.astype('S11')
	print(kmer_rows)
	print(df_rows)

	for gen in df_rows:
		if gen not in kmer_rows:
			df = df.drop([gen])

	############################################

	# For each drug
	for drug in df_cols:
		print("start: prepping amr data for ",drug)

		# Create a directory for the drugs' data
		if not os.path.exists(os.path.abspath(os.path.curdir)+'/amr_data/'+drug):
			os.mkdir(os.path.abspath(os.path.curdir)+'/amr_data/'+drug)

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

		# Mask the kmermatrix stuff
		num_rows = len(df_rows)
		mask = [1]*(num_rows)

		print("df")
		print(df)

		#print("kmer rows shape",kmer_rows.shape)


		#for i in range(kmer_rows.shape[0]):
		for i in range(len(kmer_rows)):
			x = kmer_rows[i].decode('utf-8')
			if x not in new_df_rows:
				mask[i] = 0
		bool_mask = [bool(x) for x in mask]

		print(kmer_matrix.shape)
		new_kmer_matrix = kmer_matrix[bool_mask, :]
		new_kmer_rows   = kmer_rows[bool_mask]
		print(new_kmer_matrix.shape)

		# Save the kmer row names (genomes) so that we dont have
		# to make a copy of it to manipulate it (time saver)
		np.save(os.path.abspath(os.path.curdir)+'/amr_data/'+drug+'/kmer_rows_genomes.npy', new_kmer_rows)
		# Lookup the MIC value for each genome and replace the genome
		# name with that value, and save it as a separate np array
		for i in range(new_kmer_rows.shape[0]):
			gen = new_kmer_rows[i].decode('utf-8')
			row_index = np.where(new_df_rows==gen)
			mic_val = new_df.iloc[row_index][0]
			new_kmer_rows[i] = mic_val

		joblib.dump(new_df, os.path.abspath(os.path.curdir)+'/amr_data/'+drug+'/mic_df.pkl')
		np.save(os.path.abspath(os.path.curdir)+'/amr_data/'+drug+'/kmer_matrix.npy', new_kmer_matrix)
		np.save(os.path.abspath(os.path.curdir)+'/amr_data/'+drug+'/kmer_rows_mic.npy', new_kmer_rows)
		np.save(os.path.abspath(os.path.curdir)+'/amr_data/'+drug+'/kmer_cols.npy', kmer_cols)

		print("end: prepping amr data for ",drug)


