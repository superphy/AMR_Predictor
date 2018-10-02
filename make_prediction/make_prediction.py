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


if __name__ == "__main__":
		debug_counter=0
		df = joblib.load("/amr_data/mic_class_dataframe.pkl") # Matrix of experimental MIC values
		mic_class_dict = joblib.load("/amr_data/mic_class_order_dict.pkl") # Matrix of classes for each drug
		#df_cols = df.columns
		df_cols = ['AMP']
		for drug in df_cols:
			print("\n****************",drug,"***************")
			num_classes = len(mic_class_dict[drug])
			matrix = np.load('/amr_data/'+drug+'/kmer_matrix.npy')

			sk_obj = pickle.load("make_prediction/models/"+drug+"_xgb_features.skobj")
			matrix = sk_obj.transform(matrix)

			model = pickle.load("make_prediction/models/"+drug+"_xgb_model.dat")

			prediction = model.predict(matrix)
			prediction = [int(round(float(value))) for value in prediction]
			print(prediction)
