#!/usr/bin/env python

"""
This is a script to load in a kmer matrix, its features, and its mic values
and test it against itself in a 5 fold cross validation
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
import pandas as pd
import pickle
import operator

from model_evaluators import *
from data_transformers import *

#function to load a dictionary into a pandas dataframe and scrub out irrelevant information
def create_MIC_dataframe(dictionary, drug):
	df_mic = pd.DataFrame.from_dict(dictionary, orient='index').reset_index() #create dataframe from dictionary
	df_mic = df_mic.rename(columns={'index':'MIC(mg/L)', 0:'No. Genomes'}) #rename columns
	df_mic["Drug"] = drug #add column for drug

	df_mic['MIC(mg/L)'] = df_mic['MIC(mg/L)'].replace({'<':''}, regex=True) #Strip symbols
	df_mic['MIC(mg/L)'] = df_mic['MIC(mg/L)'].replace({'>':''}, regex=True)
	df_mic['MIC(mg/L)'] = df_mic['MIC(mg/L)'].replace({'=':''}, regex=True)

	df_mic["MIC(mg/L)"] = pd.to_numeric(df_mic["MIC(mg/L)"]) #convert frequencies to float
	df_mic = df_mic.sort_values(by=['MIC(mg/L)']) #sort frequencies
	df_mic.to_pickle("data/dataframes/grdi_" + drug + "_df_mic.pkl")

if __name__ == "__main__":
	#leave at 0 features for no feature selection
	num_feats = int(sys.argv[1])
	#print("Features: ", num_feats)
	df = joblib.load("data/grdi_mic_class_dataframe.pkl") # Matrix of experimental MIC values
	mic_class_dict = joblib.load("data/grdi_mic_class_order_dict.pkl") # Matrix of classes for each drug
	df_cols = df.columns
	#df_cols = [sys.argv[2]]
	#df_cols = ["SXT"]
	for drug in df_cols:
			print("\n****************",drug,"***************")
			print("Features: ", num_feats)
			num_classes = len(mic_class_dict[drug])
			X = np.load('data/grdi_'+drug+'/kmer_matrix.npy')
			print("load shape:", X.shape)

			#create collection/dictionary of MIC frequencies to be loaded into a dataframe
			Y = np.load('data/grdi_'+drug+'/kmer_rows_mic.npy')
			if(num_feats == 1000):
				d = {}
				c = collections.Counter(Y)
				for key, value in c.items():
					d[key] = value
			for mic in mic_class_dict[drug]:
				if mic not in list(d.keys()):
					d[mic] = 0

			Z = np.load('data/grdi_'+drug+'/kmer_rows_genomes.npy')

			num_threads = 64

			cv = StratifiedKFold(n_splits=5, random_state=913824)
			cvscores = []
			window_scores = []
			mcc_scores = []
			report_scores = []
			split_counter = 0

			for train,test in cv.split(X,Y,Z):
				kmer_cols = np.load('data/grdi_unfiltered/kmer_cols.npy')
				#print("columns in initial load: ", len(kmer_cols))
				split_counter +=1
				Y[train] = encode_categories(Y[train], mic_class_dict[drug])
				Y[test]  = encode_categories(Y[test], mic_class_dict[drug])

				if(num_feats!=0):
					sk_obj = SelectKBest(f_classif, k=num_feats)
					x_train = sk_obj.fit_transform(X[train], Y[train])
					x_test  = sk_obj.transform(X[test])
					kmer_cols = kmer_cols.reshape(1, -1)
					kmer_cols = sk_obj.transform(kmer_cols)
					#print("xtrain after feat select: ", x_train.shape)
					#print("features after feat select: ", kmer_cols.shape)
				else:
					x_train = X[train]
					x_test = X[test]

				y_test = Y[test]
				y_train = Y[train]

				if drug == "TET" or drug == "SXT":
					objective = 'binary:logistic'
				else:
					objective = 'multi:softmax'

				model = XGBClassifier(learning_rate=1, n_estimators=10, objective=objective, silent=True, nthread=num_threads)
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
			result_df = pd.DataFrame(data=avg_reports, columns=['Precision', 'Recall', 'F-Score', 'Supports'])
			result_df.to_pickle("data/avg_reports/grdi_" + drug + "_df_reports.pkl")
			create_MIC_dataframe(d, drug)
			print(avg_reports)
