#!/usr/bin/env python

import numpy as np
from numpy.random import seed
from tabulate import tabulate
import pandas as pd
from decimal import Decimal

import tensorflow
from tensorflow import set_random_seed

from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential, load_model
from keras.utils import np_utils, to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import matthews_corrcoef, classification_report, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
import sys
from pandas import DataFrame
import collections


seed(913824)
set_random_seed(913824)


def eval_model(model, test_data, test_names):
    '''
    Takes a model (neural net), a set of test data, and a set of test names.
    Returns perc: the precent of correct guesses by the model using a windown of size 1.
    Returns mcc: the matthews correlation coefficient.
    Returns prediction and actual.
    '''
    # Create the prediction from the model
    prediction = model.predict_classes(test_data)

    # Reformat the true test data into the same format as the predicted data
    actual = []
    for row in range(test_names.shape[0]):
        for col in range(test_names.shape[1]):
            if(test_names[row,col]!=0):
                actual = np.append(actual,col)

    # Sum the number of correct guesses using a window: if the bin is one to either
    # side of the true bin, it is considered correct
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

    # Find the matthew's coefficient
    mcc = matthews_corrcoef(np.argmax(to_categorical(actual),axis=1),(prediction))
    return (perc, mcc, prediction, actual)

def find_major(pred, act, drug, mic_class_dict):
	class_dict = mic_class_dict[drug]
	pred = class_dict[pred]
	act  = class_dict[int(act)]
	pred = (str(pred).split("=")[-1])
	pred = ((pred).split(">")[-1])
	pred = ((pred).split("<")[-1])
	pred = int(round(float(pred)))
	act = (str(act).split("=")[-1])
	act = ((act).split(">")[-1])
	act = ((act).split("<")[-1])
	act = int(round(float(act)))

	if(drug =='AMC' or drug == 'AMP' or drug =='CHL' or drug =='FOX'):
		susc = 8
		resist = 32
	if(drug == 'AZM' or drug == 'NAL'):
		susc = 16
	if(drug == 'CIP'):
		susc = 0.06
		resist = 1
	if(drug == 'CRO'):
		susc = 1
	if(drug == 'FIS'):
		susc = 256
		resist = 512
	if(drug == 'GEN' or drug =='TET'):
		susc = 4
		resist = 16
	if(drug == 'SXT' or drug =='TIO'):
		susc = 2

	if(drug == 'AZM' or drug == 'NAL'):
		resist = 32
	if(drug == 'CRO' or drug == 'SXT'):
		resist = 4
	if(drug == 'TIO'):
		resist = 8

	if(pred <= susc and act >= resist):
		return "VeryMajorError"
	if(pred >= resist and act <= susc):
		return "MajorError"
	return "NonMajor"


def find_errors(model, test_data, test_names, genome_names, class_dict, drug, mic_class_dict):
	prediction = model.predict_classes(test_data)
	actual = []
	for row in range(test_names.shape[0]):
		for col in range(test_names.shape[1]):
			if(test_names[row,col]!=0):
				actual = np.append(actual,col)
	total_count = 0
	wrong_count = 0
	close_count = 0
	off_by_one = False
	for i in range(len(prediction)):
		total_count +=1
		pred = prediction[i]
		act = actual[i]
		if (pred == act):
			continue
		else:
			if (pred==act+1 or pred==act-1):
				close_count+=1
				off_by_one = True
			else:
				off_by_one = False
			wrong_count+=1
			print("Drug:{} Genome:{} Predicted:{} Actual:{} OBO:{} Major?:{}".format(drug, genome_names[i].decode('utf-8'), class_dict[pred], class_dict[int(act)], off_by_one, find_major(pred,act,drug,mic_class_dict)))

	print("{} out of {} were incorrect ({} were close)".format(wrong_count, total_count, close_count))


if __name__ == "__main__":
	# cce  = categorical_crossentropy
	# kull = kullback_leibler_divergence
	# poi  = poisson
	# mse  = mean_squared_error
	# msle = mean_squared_logarithmic_error
	# ch   = categorical_hinge
	# mae  = mean_absolute_error

	debug_counter=0
	df = joblib.load("amr_data/mic_class_dataframe.pkl") # Matrix of experimental MIC values
	mic_class_dict = joblib.load("amr_data/mic_class_order_dict.pkl") # Matrix of classes for each drug

	# Perform the prediction for each drug
	df_cols = df.columns
	for drug in df_cols:
		print("\n****************",drug,"***************")
		num_classes = len(mic_class_dict[drug])

		matrix = np.load('amr_data/'+drug+'/kmer_matrix.npy')
		rows_mic = np.load('amr_data/'+drug+'/kmer_rows_mic.npy')
		rows_gen = np.load('amr_data/'+drug+'/kmer_rows_genomes.npy')

		num_feats = 270
		X = SelectKBest(f_classif, k=num_feats).fit_transform(matrix, rows_mic)
		Y = rows_mic
		Z = rows_gen

		cv = StratifiedKFold(n_splits=5, random_state=913824)
		cvscores = []
		window_scores = []
		mcc_scores = []
		report_scores = []
		split_counter = 0

		#sm = SMOTE(random_state=42, k_neighbors = 1)
		#X, Y = sm.fit_sample(X, Y)
		for train,test in cv.split(X,Y, Z):
			split_counter +=1
			Y[train] = encode_categories(Y[train], mic_class_dict[drug])
			Y[test]  = encode_categories(Y[test], mic_class_dict[drug])
			y_train = to_categorical(Y[train], num_classes)
			y_test  = to_categorical(Y[test], num_classes)
			x_train = X[train]
			x_test =  X[test]

			patience = 16
			early_stop = EarlyStopping(monitor='loss', patience=patience, verbose=0, min_delta=0.005, mode='auto')
			model_save = ModelCheckpoint("best_model.hdf5",monitor='loss', verbose = 0, save_best_only =True, save_weights_only = False, mode ='auto', period =1)
			reduce_LR = ReduceLROnPlateau(monitor='loss', factor= 0.1, patience=(patience/2), verbose = 0, min_delta=0.005,mode = 'auto', cooldown=0, min_lr=0)

			model = Sequential()
			model.add(Dense(num_feats,activation='relu',input_dim=(num_feats)))
			model.add(Dropout(0.5))
			model.add(Dense(int((num_feats+num_classes)/2), activation='relu', kernel_initializer='uniform'))
			model.add(Dropout(0.5))
			model.add(Dense(num_classes, kernel_initializer='uniform', activation='softmax'))

			model.compile(loss='mean_absolute_error', metrics=['accuracy'], optimizer='adam')
			model.fit(x_train, y_train, epochs=100, verbose=0, callbacks=[early_stop, reduce_LR])

			scores = model.evaluate(x_test, y_test, verbose=0)

			results = eval_model(model, x_test, y_test)

			window_scores.append(results[0])
			mcc_scores.append(results[1])

			labels = np.arange(0,num_classes)
			report = precision_recall_fscore_support(results[3], results[2], average=None, labels=labels)
			report_scores.append(report)
			cvscores.append(scores[1] * 100)
		print("Avg base acc:   %.2f%%   (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
		print("Avg window acc: %.2f%%   (+/- %.2f%%)" % (np.mean(window_scores), np.std(window_scores)))
		print("Avg mcc:        %f (+/- %f)" % (np.mean(mcc_scores), np.std(mcc_scores)))

		np.set_printoptions(suppress=True)
		avg_reports = np.mean(report_scores,axis=0)
		avg_reports = np.transpose(avg_reports)
		avg_reports = np.around(avg_reports, decimals=2)
		col_headers = ["precision", "recall", "f1-score", "support"]
		row_headers = np.asarray(mic_class_dict[drug])
		row_headers = np.transpose(row_headers)

		print(avg_reports)
		table = tabulate(avg_reports, col_headers, tablefmt="fancy_grid")