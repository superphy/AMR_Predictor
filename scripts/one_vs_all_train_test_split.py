import numpy as np
from numpy.random import seed
import sys
import os
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN

seed(913824)

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

if __name__ == "__main__":
	#######################################################
	# Creates a total of 5 sets of training/testing data
	# for each drug. Does feature selection using given
	# number of features.
	#
	# Call with "python one_vs_all_train_test_split.py <numfeats>"
	#######################################################

	"""
	THIS SCRIPT SPLITS TO CREATE BINARY MIC CLASSES FOR ONE VS ALL CLASSIFICATION,
	see make_train_test_split.py for regular train test split
	"""

	NUM_FEATS = sys.argv[1] # default = 270

	df = joblib.load("amr_data/mic_class_dataframe.pkl") # Matrix of experimental MIC values
	mic_class_dict = joblib.load("amr_data/mic_class_order_dict.pkl") # Matrix of classes for each drug

	if not os.path.exists('./ry/'):
		os.mkdir('./ry/')

	df_cols = df.columns
	#df_cols = ['TET']
	for drug in df_cols:
		if not os.path.exists('./ry/'+drug+'/'):
			os.mkdir('./ry/'+drug+'/')


		print("\n********************",drug,"*******************")
		num_classes = len(mic_class_dict[drug])

		matrix = np.load('amr_data/'+drug+'/kmer_matrix.npy')
		rows_mic = np.load('amr_data/'+drug+'/kmer_rows_mic.npy')
		rows_gen = np.load('amr_data/'+drug+'/kmer_rows_genomes.npy')

		#X = SelectKBest(f_classif, k=int(NUM_FEATS)).fit_transform(matrix, rows_mic)
		X = matrix
		Y = rows_mic
		Z = rows_gen

		cv = StratifiedKFold(n_splits=5, random_state=913824)

		if not os.path.exists('./ry/'+drug+'/'+str(NUM_FEATS)+'feats/'):
			os.mkdir('./ry/'+drug+'/'+str(NUM_FEATS)+'feats/')



		loop = 1
		for train,test in cv.split(X,Y):

			filepath = './ry/'+drug+'/'+str(NUM_FEATS)+'feats/'

			Y[train] = encode_categories(Y[train], mic_class_dict[drug])
			Y[test]  = encode_categories(Y[test], mic_class_dict[drug])

			y_train = np.array([], dtype = 'i4')

			#set special class to whatever you are trying to predict, if you want to compare class 0 vs the rest,
			#set special class to 0
			special_class = 0
			for i in Y[train]:
				i = int(i.decode('utf-8'))
				if(i=special_class):
					y_train = np.append(y_train, 1)
				else:
					y_train = np.append(y_train, 0)

			y_test = np.array([],dtype = 'i4')
			for i in Y[test]:
				i = int(i.decode('utf-8'))
				if(i=special_class):
					y_test = np.append(y_test, 1)
				else:
					y_test = np.append(y_test, 0)

			#feature selection
			sk_obj = SelectKBest(f_classif, k=int(NUM_FEATS))
			x_train = sk_obj.fit_transform(X[train], y_train)
			x_test  = sk_obj.transform(X[test])

			print(x_train.shape)
			print(y_train.shape)
			print(x_test.shape)
			print(y_test.shape)

			filepath=filepath+'fold'+str(loop)+'/'
			if not os.path.exists(filepath):
				os.mkdir(filepath)

			np.save(filepath+'x_train.npy', x_train)
			np.save(filepath+'x_test.npy', x_test)
			np.save(filepath+'y_train.npy', y_train)
			np.save(filepath+'y_test.npy', y_test)
			#np.save(filepath+'genome_train.npy', Z[train])
			#np.save(filepath+'genome_test.npy', Z[test])

			loop+=1
