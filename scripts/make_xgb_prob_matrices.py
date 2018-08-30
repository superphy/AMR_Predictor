#!/usr/bin/env python

import numpy as np
from numpy.random import seed

import pandas as pd
from pandas import DataFrame
from decimal import Decimal
from xgboost import XGBClassifier
import sys
import os
import pickle

import tensorflow
from tensorflow import set_random_seed

#from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential#, load_model
from keras.utils import np_utils, to_categorical
#from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import matthews_corrcoef, classification_report, precision_recall_fscore_support, confusion_matrix

from hpsklearn import HyperoptEstimator, svc, xgboost_classification
from hyperopt import tpe

import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
	##################################################################
	# call with
	#	time python make_xgb_prob_matrices.py <numfeats> <drug> <fold>
	# to do all folds
	#	for i in {1..5}; do python make_xgb_prob_matrices.py <numfeats> <drug> '$i'; done
	# to do all folds on wafflesnt
	#	sbatch -c 16 --mem 80G --partition NMLResearch --wrap='for i in {1..5}; do python make_xgb_prob_matrices.py <numfeats> <drug> "$i"; done'
	##################################################################

	feats = sys.argv[1]
	drug = sys.argv[2]
	fold = sys.argv[3]

	# Useful to have in the slurm output
	print("************************************")
	print("make_xgb_prob_matrices.py")
	print(drug, feats, fold)
	print("************************************")

	# Load data
	mic_class_dict = joblib.load(os.path.abspath(os.path.curdir)+"/amr_data/mic_class_order_dict.pkl")
	class_dict = mic_class_dict[drug]
	num_classes = len(mic_class_dict[drug])
	filepath = os.path.abspath(os.path.curdir)+'/amr_data/'+drug+'/'+str(feats)+'feats/fold'+str(fold)+'/'
	genome_names = np.load(filepath+'genome_test.npy')

	# Load training and testing sets
	#x_train = np.load(filepath+'x_train.npy')
	x_test  = np.load(filepath+'x_test.npy')
	#y_train = np.load(filepath+'y_train.npy')
	y_test  = np.load(filepath+'y_test.npy')
	# Convert to relevant types for the neural net
	unencoded_y_test = y_test
	#y_train = y_train.astype('S11')
	#y_train = to_categorical(y_train, num_classes)
	y_test  = to_categorical(y_test, num_classes)

	## Model #######################################################
	model = pickle.load(open(filepath+'xgb_model.dat', "rb"))
	#load_model(filepath+'xgb_model.dat')
	################################################################

	first_guess = 0
	second_guess = 0
	third_guess = 0
	fourth_guess = 0

	#print(model)
	#model = model.best_model()

	model = model._best_learner

	pred = model.predict_proba(x_test)
	print(x_test.shape)
	print(pred.shape)
	print('num_classes', num_classes)

	unencoded_y_test = [int(val.decode('utf-8')) for val in unencoded_y_test]
	counter = 0
	"""for i in range(len(sig_pred)):
		if(sig_pred[i]==unencoded_y_test[i]):
			sig_counter +=1
			first_guess"""
	prob_matrix = np.zeros((x_test.shape[0],4))
	import operator
	for i in range(len(pred)):
		row_dict = {}
		for j in range(pred.shape[1]):
			row_dict[j]=pred[i,j]
		first_g = (max(row_dict.items(), key=operator.itemgetter(1))[0])
		prob_matrix[i,0] = row_dict[first_g]
		row_dict[first_g] = 0
		sec_g =(max(row_dict.items(), key=operator.itemgetter(1))[0])
		prob_matrix[i,1] = row_dict[sec_g]
		row_dict[sec_g] = 0
		third_g =(max(row_dict.items(), key=operator.itemgetter(1))[0])
		prob_matrix[i,2] = row_dict[third_g]
		row_dict[third_g] = 0
		fourth_g =(max(row_dict.items(), key=operator.itemgetter(1))[0])
		if(unencoded_y_test[i]==first_g):
			counter+=1
			first_guess +=1
			prob_matrix[i,3] = 1
		elif(unencoded_y_test[i]==sec_g):
			second_guess +=1
			prob_matrix[i,3] = 2
		elif(unencoded_y_test[i]==third_g):
			third_guess +=1
			prob_matrix[i,3] = 3
		elif(unencoded_y_test[i]==fourth_g):
			fourth_guess +=1
		#add first prob, first+2, and 1+2+3 to array, and which class was right

	print(prob_matrix)

	size = len(pred)
	print("1st Guess: {}, 2nd Guess: {}, 3rd Guess: {}, 4th Guess: {}".format(first_guess/size, second_guess/size, third_guess/size, fourth_guess/size))

	## Score #######################################################
	#score = model.evaluate(x_test, y_test, verbose=0)
	#score_1d = eval_modelOBO(model, x_test, y_test)
	#print("Base: {} 1d: {}".format(score[1], score_1d[0]))

	np.save(filepath+'xgb_prob_matrix.npy', prob_matrix)

	test = np.load(filepath+"/xgb_prob_matrix.npy")

	print(test)

	#with open(filepath+'xgb_prob_out.txt','w') as f:
	#	f.write("1st Guess: {}, 2nd Guess: {}, 3rd Guess: {}, 4th Guess: {}".format(first_guess/size, second_guess/size, third_guess/size, fourth_guess/size))
	#	f.write("Base: {} 1d: {}".format(score[1], score_1d[0]))