
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

from model_evaluators import *
from data_transformers import *

df = joblib.load("data/public_mic_class_dataframe.pkl") # Matrix of experimental MIC values
mic_class_dict = joblib.load("data/public_mic_class_order_dict.pkl") # Matrix of classes for each drug
df_cols = df.columns
df_cols=['AMP']
for drug in df_cols:
		print("\n****************",drug,"***************")
		num_classes = len(mic_class_dict[drug])
		X = np.load('data/'+drug+'/kmer_matrix.npy')
		print("load shape:", X.shape)
		Y = np.load('data/'+drug+'/kmer_rows_mic.npy')
		Z = np.load('data/'+drug+'/kmer_rows_genomes.npy')

		#leave at 0 for no feature selection
		num_feats = 0
		num_threads = 144

		cv = StratifiedKFold(n_splits=5, random_state=913824)
		cvscores = []
		window_scores = []
		mcc_scores = []
		report_scores = []
		split_counter = 0

		for train,test in cv.split(X,Y,Z):
			split_counter +=1
			Y[train] = encode_categories(Y[train], mic_class_dict[drug])
			Y[test]  = encode_categories(Y[test], mic_class_dict[drug])

			if(num_feats!=0):
				sk_obj = SelectKBest(f_classif, k=num_feats)
				X[train] = sk_obj.fit_transform(X[train], Y[train])
				X[test]  = sk_obj.transform(X[test])

			model = XGBClassifier(learning_rate=1, n_estimators=10, objective='multi:softmax', silent=True, nthread=num_threads)
			model.fit(X[train],Y[train])

			results = xgb_tester(model, X[test], Y[test], 0)
			OBOResults = xgb_tester(model, X[test], Y[test], 1)

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
