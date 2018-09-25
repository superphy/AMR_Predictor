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

from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential#, load_model
from keras.utils import np_utils, to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import matthews_corrcoef, classification_report, precision_recall_fscore_support, confusion_matrix

from hpsklearn import HyperoptEstimator, svc, xgboost_classification
from hyperopt import tpe

import warnings
warnings.filterwarnings('ignore')


seed(913824)
set_random_seed(913824)


def eval_modelOBO(model, test_data, test_names):
    '''
    Takes a model (neural net), a set of test data, and a set of test names.
    Returns perc: the precent of correct guesses by the model within 1 dilution.
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

    # Sum the number of correct guesses within 1 dilution: if the bin is one to either
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

	if not os.path.exists(os.path.abspath(os.path.curdir)+'/amr_data/errors'):
		os.mkdir(os.path.abspath(os.path.curdir)+'/amr_data/errors')

	err_file = open(os.path.abspath(os.path.curdir)+'/amr_data/errors/'+str(sys.argv[1])+'_feats_NN_errors.txt', 'a+')

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
			err_file.write("Drug:{} Genome:{} Predicted:{} Actual:{} OBO:{} Major?:{}\n".format(drug, genome_names[i], class_dict[pred], class_dict[int(act)], off_by_one, find_major(pred,act,drug,mic_class_dict)))


if __name__ == "__main__":
	##################################################################
	# call with
	#	time python roary_NN.py <numfeats> <drug> <fold>
	# to do all folds
	#	for i in {1..5}; do python roary_NN.py <numfeats> <drug> '$i'; done
	# to do all folds on waffles
	#	sbatch -c 16 --mem 80G --partition NMLResearch --wrap='for i in {1..5}; do python roary_NN.py <numfeats> <drug> "$i"; done'
	# OR
	#   or use roary_NN.snake (change the features num)
	##################################################################

	feats = sys.argv[1]
	drug = sys.argv[2]
	fold = sys.argv[3]

	# Useful to have in the slurm output
	print("************************************")
	print("roary_NN.py")
	print(drug, feats, fold)
	print("************************************")

	## Load data ###################################################
	filepath = os.path.abspath(os.path.curdir)+'/roary_amr/'+drug+'/'+str(feats)+'feats/fold'+str(fold)+'/'

	mic_class_dict = joblib.load("amr_data/mic_class_order_dict.pkl") # Matrix of classes for each drug
	num_classes = len(mic_class_dict[drug])
	genome_names = np.load(filepath+'/genome_test.npy')

	matrix = np.load('roary_amr/'+drug+'/matrix.npy')
	rows_mic = np.load('roary_amr/'+drug+'/matrix_rows_mic.npy')
	rows_gen = np.load('roary_amr/'+drug+'/matrix_rows_genomes.npy')

	x_train = np.load(filepath+'x_train.npy')
	x_test  = np.load(filepath+'x_test.npy')
	y_train = np.load(filepath+'y_train.npy')
	y_test  = np.load(filepath+'y_test.npy')
	y_train=y_train.astype('S11')
	y_train = to_categorical(y_train, num_classes)
	y_test  = to_categorical(y_test, num_classes)
	################################################################

	## Model #######################################################
	patience = 16
	early_stop = EarlyStopping(monitor='loss', patience=patience, verbose=0, min_delta=0.005, mode='auto')
	model_save = ModelCheckpoint("best_model.hdf5",monitor='loss', verbose = 0, save_best_only =True, save_weights_only = False, mode ='auto', period =1)
	reduce_LR = ReduceLROnPlateau(monitor='loss', factor= 0.1, patience=(patience/2), verbose = 0, min_delta=0.005,mode = 'auto', cooldown=0, min_lr=0)
	
	model = Sequential()
	model.add(Dense(x_train.shape[1],activation='relu',input_dim=(x_train.shape[1])))
	model.add(Dense(num_classes, kernel_initializer='uniform', activation='softmax'))
	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
	model.fit(x_train, y_train, epochs=1, verbose=1, callbacks=[early_stop, reduce_LR])
	################################################################

	# Find and record errors
	find_errors(model, x_test, y_test, genome_names, class_dict, drug, mic_class_dict)

	## Score #######################################################
	score = model.evaluate(x_test, y_test, verbose=0)
	score_1d = eval_modelOBO(model, x_test, y_test)
	y_true = score_1d[3]
	y_pred = score_1d[2]
	sc = {'base acc': [score[1]], '1d acc': [score_1d[0]], 'mcc':[score_1d[1]]}
	score_df = DataFrame(sc)
	################################################################

	## Confusion Matrix ############################################
	labels = np.arange(0,num_classes)
	conf = confusion_matrix(y_true, y_pred, labels=labels)
	conf_df = DataFrame(conf, index=mic_class_dict[drug]) # Turn the results into a pandas dataframe (df)
	conf_df.set_axis(mic_class_dict[drug], axis='columns', inplace=True) # Label the axis
	################################################################

	## Classification Report #######################################
	report = classification_report(y_true, y_pred, target_names=mic_class_dict[drug])
	rep_df = metrics_report_to_df(y_true, y_pred)
	################################################################

	## Save Everything #############################################
	model.save(filepath+'/nn_model.hdf5')
	conf_df.to_pickle(filepath+'nn_conf_df.pkl')
	score_df.to_pickle(filepath+'nn_score_df.pkl')
	rep_df.to_pickle(filepath+'nn_rep_df.pkl')
	with open(filepath+'nn_out.txt','w') as f:
		f.write("\nBase acc: {0}%\n".format(score[0]))
		f.write("1-d acc: {0}%\n".format(score_1d[0]))
		f.write("MCC: {0}\n".format(round(score_1d[1],4)))
		f.write("\nConfusion Matrix\n{0}\n".format(conf_df))		
		f.write("\nClassification Report\n{0}\n".format(report))
		f.write("Best performing model chosen hyper-parameters:\n{0}".format(model))
	################################################################

