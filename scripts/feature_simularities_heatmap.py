#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest, f_classif
import concurrent.futures
import os
import sys

def encode_categories(data, class_dict):
	'''
	Given a set of bin numbers (data), and a set of classes (class_dict),
	translates the classes into bins.
	Eg. goes from ['<=1,2,>=4'] into [0,1,2]
	'''
	arry = np.array([], dtype = 'i4')
	for item in data:
		temp = str(item)
		temp = int(''.join(filter(str.isdigit, temp)))
		for index in range(len(class_dict)):
			check = class_dict[index]
			check = int(''.join(filter(str.isdigit, check)))
			if temp == check:
				temp = index
		arry = np.append(arry,temp)
	return arry

def same_distro(col1, col2):
	p = 0.01
	#return (stats.ks_2samp(kmer_matrix[0,:], kmer_matrix[1,:])[1])<p
	return 1

def exact_same(col1, col2):
	for i in range(len(col1)):
		if col1[i]!=col2[i]:
			return 0
	return 1

def num_of_non_same(col1, col2):
	counter = 0
	for i in range(len(col1)):
		if (col1[i]!=col2[i]):
			counter +=1
	return counter

if __name__ == "__main__":
	df = joblib.load((os.path.abspath(os.path.curdir)+"/non_grdi/amr_data/mic_class_dataframe.pkl")) # Matrix of experimental MIC values
	mic_class_dict = joblib.load((os.path.abspath(os.path.curdir)+"/non_grdi/amr_data/mic_class_order_dict.pkl")) # Matrix of classes for each drug

	#drugs = ["AMP","AMC","AZM","CHL","CIP","CRO","FIS","FOX","GEN","NAL","SXT","TET","TIO"]
	drugs = [sys.argv[1]]

	for drug in drugs:

		print("*******",drug,"*******")
		kmer_matrix = np.load((os.path.abspath(os.path.curdir)+'/non_grdi/amr_data/'+drug+'/kmer_matrix.npy'))
		kmer_cols = np.load((os.path.abspath(os.path.curdir)+'/non_grdi/amr_data/'+drug+'/kmer_cols.npy'))
		kmer_rows_mic = np.load((os.path.abspath(os.path.curdir)+'/non_grdi/amr_data/'+drug+'/kmer_rows_mic.npy'))

		kmer_rows_mic = encode_categories(kmer_rows_mic, mic_class_dict[drug])

		#sk_obj = SelectKBest(f_classif, k=1000)
		#kmer_matrix = sk_obj.fit_transform(kmer_matrix, kmer_rows_mic)

		fvals, pvals = f_classif(kmer_matrix, kmer_rows_mic)

		# use this loop to find a good cutoff for later

		for i in range(50):
			pcutoff = 10**(-i)
			fmask = pvals < pcutoff
			print(pcutoff, '->', np.sum(fmask))
			if(np.sum(fmask) < 30000):
				break

		"""
		pcutoff = 10**(-43)
		if(drug == 'AZM'):
			pcutoff = 10**(-25)
		elif(drug == 'CIP'):
			pcutoff = 10**(-32)
		elif(drug == 'NAL'):
			pcutoff = 10**(-37)
		elif(drug == 'GEN'):
			pcutoff = 10**(-39)
		elif(drug == 'SXT'):
			pcutoff = 10**(-21)
		"""
		pcutoff = pcutoff*10
		fmask = pvals < pcutoff
		print(pcutoff)
		print('fmask:', np.sum(fmask))
		#print("Before:", kmer_matrix.shape)

		kmer_matrix = kmer_matrix[:,fmask]
		kmer_cols = kmer_cols[fmask]

		#kmer_matrix = [[1,2,2,3,3],[2,2,2,3,3],[2,2,2,3,3]]
		#print("After:", kmer_matrix.shape)
		#kmer_matrix = np.asarray(kmer_matrix)
		#print(kmer_matrix)
		heatmap_matrix = np.zeros((kmer_matrix.shape[1], kmer_matrix.shape[1]), dtype='uint32')
		kmer_matrix = kmer_matrix.transpose()
		copy_mask = np.zeros(kmer_matrix.shape[0])
		copy_count = 0
		break_counter = 0
		for cnti, i in enumerate(kmer_matrix):
			#print(i)
			for cntj, j in enumerate(kmer_matrix):
				if (cnti<cntj):
					heatmap_matrix[cnti][cntj]=num_of_non_same(i,j)
			#break_counter+=1
			#if(break_counter==10):
				#break
		np.save('no_dup_feats/'+drug+'_heatmap_matrix.npy', heatmap_matrix)
		np.save('no_dup_feats/'+drug+'_heatmap_kmer_cols.npy',kmer_cols)
