#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest, f_classif
import concurrent.futures
import os

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

if __name__ == "__main__":
	df = joblib.load((os.path.abspath(os.path.curdir)+"/non_grdi/amr_data/mic_class_dataframe.pkl")) # Matrix of experimental MIC values
	mic_class_dict = joblib.load((os.path.abspath(os.path.curdir)+"/non_grdi/amr_data/mic_class_order_dict.pkl")) # Matrix of classes for each drug

	drugs = ["AMP","AMC","AZM","CHL","CIP","CRO","FIS","FOX","GEN","NAL","SXT","TET","TIO"]
	#drugs = ['AMP']

	for drug in drugs:
		print("*******",drug,"*******")
		kmer_matrix = np.load((os.path.abspath(os.path.curdir)+'/non_grdi/amr_data/'+drug+'/kmer_matrix.npy'))
		kmer_cols = np.load((os.path.abspath(os.path.curdir)+'/non_grdi/amr_data/'+drug+'/kmer_cols.npy'))
		kmer_rows_mic = np.load((os.path.abspath(os.path.curdir)+'/non_grdi/amr_data/'+drug+'/kmer_rows_mic.npy'))

		kmer_rows_mic = encode_categories(kmer_rows_mic, mic_class_dict[drug])

		#sk_obj = SelectKBest(f_classif, k=1000)
		#kmer_matrix = sk_obj.fit_transform(kmer_matrix, kmer_rows_mic)

		fvals, pvals = f_classif(kmer_matrix, kmer_rows_mic)
		for i in range(50):
			pcutoff = 10**(-i)
			fmask = pvals < pcutoff
			print(pcutoff, '->', np.sum(fmask))

		#kmer_matrix.loc[:,fmask]

		#kmer_matrix = kmer_matrix.transpose()
		#no_dup_kmer_matrix  = np.vstack({tuple(row) for row in arr1})
		#no_dup_kmer_matrix = no_dup_kmer_matrix.transpose()
		#print(no_dup_kmer_matrix.shape)
		#kmer_matrix = kmer_matrix.transpose()
		#with concurrent.futures.ThreadPoolExecutor(max_workers = 31) as executor:
		#   print(len(list(executor.map(exact_same, kmer_matrix, kmer_matrix))))

		#print(stats.ks_2samp(kmer_matrix[0,:], kmer_matrix[1,:]))
