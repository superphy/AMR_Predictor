#!/usr/bin/env python

"""
This is a script to load in a kmer matrix, its features, and its mic values
and test it against the grdi dataset
"""
import numpy as np
from xgboost import XGBClassifier
from keras.utils import np_utils, to_categorical
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import matthews_corrcoef, classification_report, precision_recall_fscore_support
import collections
from sklearn.externals import joblib
import sys

from model_evaluators import *
from data_transformers import *

if __name__ == "__main__":
	#leave at 0 features for no feature selection
	num_feats = int(sys.argv[1])
	#print("Features: ", num_feats)
	df = joblib.load("data/public_mic_class_dataframe.pkl") # Matrix of experimental MIC values
	mic_class_dict = joblib.load("data/public_mic_class_order_dict.pkl") # Matrix of classes for each drug
	df_cols = df.columns
	df_cols = [sys.argv[2]]
	for drug in df_cols:
			print("\n****************",drug,"***************")
			print("Features: ", num_feats)
			num_classes = len(mic_class_dict[drug])

			pub_matrix = np.load('data/'+drug+'/kmer_matrix.npy')
			pub_rows_mic = np.load('data/'+drug+'/kmer_rows_mic.npy')
			pub_rows_genomes = np.load('data/'+drug+'/kmer_rows_genomes.npy')

			grdi_matrix = np.load('data/grdi_'+drug+'/kmer_matrix.npy')
			grdi_rows_mic = np.load('data/grdi_'+drug+'/kmer_rows_mic.npy')
			grdi_rows_genomes = np.load('data/grdi_'+drug+'/kmer_rows_genomes.npy')

			num_threads = 64

			cv = StratifiedKFold(n_splits=5, random_state=913824)
			cvscores = []
			window_scores = []
			mcc_scores = []
			report_scores = []
			split_counter = 0

			#for train,test in cv.split(X,Y,Z):
			x_train = grdi_matrix
			y_train = grdi_rows_mic
			x_test  = pub_matrix
			y_test  = pub_rows_mic

			#kmer_cols = np.load('data/unfiltered/kmer_cols.npy')
			#print("columns in initial load: ", len(kmer_cols))
			split_counter +=1
			y_train = encode_categories(y_train, mic_class_dict[drug])
			y_test  = encode_categories(y_test, mic_class_dict[drug])

			if(num_feats!=0):
				sk_obj = SelectKBest(f_classif, k=num_feats)
				x_train = sk_obj.fit_transform(x_train, y_train)
				x_test  = sk_obj.transform(x_test)
				#kmer_cols = kmer_cols.reshape(1, -1)
				#kmer_cols = sk_obj.transform(kmer_cols)
				#print("xtrain after feat select: ", x_train.shape)
				#print("features after feat select: ", kmer_cols.shape)

			model = XGBClassifier(learning_rate=1, n_estimators=10, objective='multi:softmax', silent=True, nthread=num_threads)
			model.fit(x_train,y_train)

			#feat_array = np.asarray(model.feature_importances_)
			#sort_feat_array = sorted(feat_array)
			#sort_feat_array.reverse()
			#fifth_largest = sort_feat_array[4]
			#top_five_mask = [i>=fifth_largest for i in feat_array]
			#print("Top 5: ", kmer_cols[:,top_five_mask])
			#print("Top 5: ", feat_array[top_five_mask])

			results = xgb_tester(model, x_test, y_test, 0)
			OBOResults = xgb_tester(model, x_test, y_test, 1)

			window_scores.append(OBOResults[0])
			mcc_scores.append(results[1])

			labels = np.arange(0,num_classes)
			report = precision_recall_fscore_support(results[3], results[2], average=None, labels=labels)
			report_scores.append(report)
			cvscores.append(results[0])

			print("Avg base acc:   %.2f%%   (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
			print("Avg window acc: %.2f%%   (+/- %.2f%%)" % (np.mean(window_scores), np.std(window_scores)))
			print("Avg mcc:        %f (+/- %f)" % (np.mean(mcc_scores), np.std(mcc_scores)))

			np.set_printoptions(suppress=True)
			avg_reports = np.mean(report_scores,axis=0)
			avg_reports = np.transpose(avg_reports)
			avg_reports = np.around(avg_reports, decimals=2)
			print(avg_reports)
