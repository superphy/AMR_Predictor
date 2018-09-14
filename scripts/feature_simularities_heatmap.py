#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest, f_classif
import concurrent.futures
import os

"""
This program takes in a 2d matrix, extracts the best features based on pvalue from f_classif,
and returns another 2d matrix containing the number of differences between each set of features,
 example output:
	col1 col2 col3
col1  0	   3    5
col2  3    0    7
col3  5    7    0


Matrix for testing:
Kmer_matrix = np.asarray([[1,2,2,3,3],
						  [2,2,2,3,3],
						  [2,2,2,3,3]])
"""
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

def num_of_non_same(cols):
	counter = 0
	col1, col2 = cols
	#print(col1, col2)
	col1 = T_kmer_matrix[col1]
	col2 = T_kmer_matrix[col2]
	for i in range(len(col1)):
		if (col1[i]!=col2[i]):
			counter +=1
	return counter

def exact_same(cols):
	col1, col2 = cols
	#print(col1, col2)
	col1 = T_kmer_matrix[col1]
	col2 = T_kmer_matrix[col2]
	#print("in exact_same with:", col1, col2)
	for i in range(len(col1)):
		if col1[i]!=col2[i]:
			return 0
	return 1

drug = 'AMP'
df = joblib.load((os.path.abspath(os.path.curdir)+"/non_grdi/amr_data/mic_class_dataframe.pkl")) # Matrix of experimental MIC values
mic_class_dict = joblib.load((os.path.abspath(os.path.curdir)+"/non_grdi/amr_data/mic_class_order_dict.pkl")) # Matrix of classes for each drug
kmer_matrix = np.load((os.path.abspath(os.path.curdir)+'/non_grdi/amr_data/'+drug+'/kmer_matrix.npy'))
kmer_cols = np.load((os.path.abspath(os.path.curdir)+'/non_grdi/amr_data/'+drug+'/kmer_cols.npy'))
kmer_rows_mic = np.load((os.path.abspath(os.path.curdir)+'/non_grdi/amr_data/'+drug+'/kmer_rows_mic.npy'))
kmer_rows_mic = encode_categories(kmer_rows_mic, mic_class_dict[drug])
fvals, pvals = f_classif(kmer_matrix, kmer_rows_mic)
pcutoff = 10**(-45)
fmask = pvals < pcutoff
kmer_matrix = kmer_matrix[:,fmask]
kmer_cols = kmer_cols[fmask]

T_kmer_matrix = kmer_matrix.transpose()



heatmap_matrix=[]
#Single row of comparasions, creates an array of tuples for passing into concurrent.futures
srofc = np.zeros((T_kmer_matrix.shape[0],2), dtype = 'uint8')
for i in range(T_kmer_matrix.shape[0]):
	srofc[i][1] = i
for row in range(T_kmer_matrix.shape[0]):
	for j in range(T_kmer_matrix.shape[0]):
		srofc[j][0] = row
	with concurrent.futures.ThreadPoolExecutor(max_workers = 64) as executor:
		heatmap_matrix.append(list(executor.map(num_of_non_same, srofc)))

np.save('no_dup_feats/'+drug+'_par_heatmap_matrix.npy', np.asarray(heatmap_matrix))
np.save('no_dup_feats/'+drug+'_par_heatmap_kmer_cols.npy',kmer_cols)
