#!/usr/bin/env python

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import pickle
import os
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import matthews_corrcoef, classification_report, precision_recall_fscore_support
import sys
import itertools
from decimal import Decimal
from keras.utils import np_utils, to_categorical


def decode_categories(data, class_dict):
	'''
	Given a set of bin numbers (data), and a set of classes (class_dict),
	translates the bins into classes.
	Eg. goes from [0,1,2] into ['<=1,2,>=4']
	'''
	arry = np.array([])
	for item in data:
		arry = np.append(arry,class_dict[item])
	return arry


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

def find_index(kmer):
	for i, element in enumerate(all_feats):
		if kmer == element:
			return i

def eval_model(model, test_data, test_names):
	'''
	Takes a model (neural net), a set of test data, and a set of test names.
	Returns perc: the precent of correct guesses by the model using a windown of size 1.
	Returns mcc: the matthews correlation coefficient.
	Returns prediction and actual.
	'''
	# Create and save the prediction from the model
	prediction = model.predict(test_data)
	prediction = [int(round(float(value))) for value in prediction]
	#np.save('prediction.npy', prediction)

	actual = test_names
	actual = [int(float(value)) for value in actual]
	# Sum the number of correct guesses using a window: if the bin is one to either
	# side of the true bin, it is considered correct
	total_count = 0
	correct_count = 0
	for i in range(len(prediction)):
		total_count +=1
		pred = prediction[i]
		act = actual[i]
		if pred==act:
			correct_count+=1
	# Calculate the percent of correct guesses
	perc = (correct_count*100)/total_count
	perc = Decimal(perc)
	perc = round(perc,2)

	#print("When allowing the model to guess MIC values that are next to the correct value:")
	#print("This model correctly predicted mic values for {} out of {} genomes ({}%).".format(correct_count,total_count,perc))
	#print("\nMCC: ", matthews_corrcoef(np.argmax(to_categorical(actual),axis=1),(prediction)))

	# Find the matthew's coefficient
	mcc = matthews_corrcoef(np.argmax(to_categorical(actual),axis=1),(prediction))
	return (perc, mcc, prediction, actual)

def eval_modelOBO(model, test_data, test_names):
	'''
	Takes a model (neural net), a set of test data, and a set of test names.
	Returns perc: the precent of correct guesses by the model using a windown of size 1.
	Returns mcc: the matthews correlation coefficient.
	Returns prediction and actual.
	'''
	# Create and save the prediction from the model
	prediction = model.predict(test_data)
	prediction = [int(round(float(value))) for value in prediction]
	#np.save('prediction.npy', prediction)

	actual = test_names
	actual = [int(float(value)) for value in actual]
	# Sum the number of correct guesses using a window: if the bin is one to either
	# side of the true bin, it is considered correct
	#print("actual: ", actual)
	#print("predict: ", prediction)
	total_count = 0
	correct_count = 0
	for i in range(len(prediction)):
		total_count +=1
		pred = prediction[i]
		act = actual[i]
		if pred==act or pred==act+1 or pred==act-1:
			correct_count+=1
	# Calculate the percent of correct guesses
	perc = (correct_count*100)/total_count
	perc = Decimal(perc)
	perc = round(perc,2)

	#print("When allowing the model to guess MIC values that are next to the correct value:")
	#print("This model correctly predicted mic values for {} out of {} genomes ({}%).".format(correct_count,total_count,perc))
	#print("\nMCC: ", matthews_corrcoef(np.argmax(to_categorical(actual),axis=1),(prediction)))

	# Find the matthew's coefficient
	#mcc = matthews_corrcoef(np.argmax(to_categorical(actual),axis=1),(prediction))
	mcc = 1
	return (perc, mcc, prediction, actual)

if __name__ == "__main__":
	df = joblib.load("non_grdi/amr_data/mic_class_dataframe.pkl") # Matrix of experimental MIC values
	mic_class_dict = joblib.load("non_grdi/amr_data/mic_class_order_dict.pkl") # Matrix of classes for each drug
	#df_cols = df.columns
	df_cols = ['AMP']
	for drug in df_cols:
			print("\n****************",drug,"***************")
			num_classes = len(mic_class_dict[drug])
			matrix = np.load('non_grdi/amr_data/'+drug+'/kmer_matrix.npy')
			kmer_rows_mic = np.load('non_grdi/amr_data/'+drug+'/kmer_rows_mic.npy')
			"""
			with open("make_prediction/models/"+drug+"_xgb_features.skobj", 'rb') as sk_handle:
					sk_obj = pickle.load(sk_handle)
			matrix = sk_obj.transform(matrix)
			"""

			#generating a list of all possible kmer sequences
			chars = 'AGCT'
			count = 11
			size = 4**count
			all_feats = np.empty(size,dtype='object')
			for i, item in enumerate(itertools.product(chars, repeat = count)):
				all_feats[i]=("".join(item))

			#converting bytes to strings
			all_feats = [str(i) for i in all_feats]

			#loading the indexes of the top 270 features found when making the model
			topf = np.load("make_prediction/models/"+drug+"_xgb_features.npy")

			#generating a mask to apply to the matrix to reduce it to the correct feature count
			feat_mask = np.zeros(size)
			topf = [all_feats[i] for i in topf]
			for i, element in enumerate(all_feats):
				if(element in topf):
					feat_mask[i] = 1
			feat_mask = [i==1 for i in feat_mask]

						
			assert(np.sum(feat_mask)==270 and len(topf)==270)
			#loading the model
			with open('make_prediction/models/'+drug+'_xgb_model.dat', 'rb') as model_handle:
			#with open('/Drives/L/Bioinformatics-Lethbridge/salmonella_data/jan/skmer/amr_data/AMP/270feats/fold3/xgb_model.dat', 'rb') as model_handle:
			model = pickle.load(model_handle)
			matrix = matrix.transpose()
			#applying the feature mask
			matrix = matrix[feat_mask]
			matrix = matrix.transpose()
			kmer_rows_mic = encode_categories(kmer_rows_mic, mic_class_dict[drug])
			print(eval_model(model,matrix, kmer_rows_mic)[0])
			print(eval_modelOBO(model, matrix, kmer_rows_mic)[0])
			#prediction = model.predict(matrix)
			#prediction = [int(round(float(value))) for value in prediction]

