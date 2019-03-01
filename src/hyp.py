#!/usr/bin/env python

import numpy as np
from numpy.random import seed
import pandas as pd
from pandas import DataFrame
import sys
import pickle
from decimal import Decimal
import os

import tensorflow
from tensorflow import set_random_seed

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

from hyperopt import Trials, STATUS_OK, tpe
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Flatten, BatchNormalization
from keras.models import Sequential, load_model
from keras.utils import np_utils, to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

from sklearn import metrics
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import matthews_corrcoef, confusion_matrix, classification_report


seed(913824)
set_random_seed(913824)

def eval_model(model, test_data, test_names):
	'''
	Takes a model (neural net), a set of test data, and a set of test names.
	Returns perc: the precent of correct guesses by the model within one dilution.
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

	err_file = open(os.path.abspath(os.path.curdir)+'/amr_data/errors/'+str(sys.argv[1])+'_feats_hyperas_errors.txt', 'a+')

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
			err_file.write("Drug:{} Genome:{} Predicted:{} Actual:{} OBO:{} Major?:{}\n".format(drug, genome_names[i].decode('utf-8'), class_dict[pred], class_dict[int(act)], off_by_one, find_major(pred,act,drug,mic_class_dict)))


def data():
	from keras.utils import to_categorical

	# Matrix of classes for each drug
	mic_class_dict = joblib.load(os.path.abspath(os.path.curdir)+"/data/public_mic_class_order_dict.pkl")

	drug = sys.argv[2]
	num_classes = len(mic_class_dict[drug])

	#filepath = os.path.abspath(os.path.curdir)+'/amr_data/'+drug+'/'+str(sys.argv[1])+'feats/fold'+str(sys.argv[3])+'/'
	x_train = np.load('x_train.npy')
	x_test  = np.load('x_test.npy')
	y_train = np.load('y_train.npy')
	y_test  = np.load('y_test.npy')

	y_train = to_categorical(y_train, num_classes)
	y_test  = to_categorical(y_test, num_classes)

	return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
	patience = {{choice([4,8,12,16])}}
	early_stop = EarlyStopping(monitor='loss', patience=patience, verbose=0, min_delta=0.005, mode='auto')
	model_save = ModelCheckpoint("best_model.hdf5",monitor='loss', verbose = 0, save_best_only =True, save_weights_only = False, mode ='auto', period =1)
	reduce_LR = ReduceLROnPlateau(monitor='loss', factor= 0.1, patience=(patience/2), verbose = 0, min_delta=0.005,mode = 'auto', cooldown=0, min_lr=0)

	model = Sequential()

	model.add(Dense(x_train.shape[1],activation='relu',input_dim=(x_train.shape[1])))
	model.add(Dropout({{uniform(0,1)}}))

	num_layers = {{choice(['zero', 'one', 'two', 'three', 'four', 'five'])}}

	if (num_layers in ['one','two','three','four','five']):
		model.add(Dense(int({{uniform(num_classes,x_train.shape[1])}})))
		model.add(Dropout({{uniform(0,1)}}))
	if (num_layers in ['two','three','four','five']):
		model.add(Dense(int({{uniform(num_classes,x_train.shape[1])}})))
		model.add(Dropout({{uniform(0,1)}}))
	if (num_layers in ['three','four','five']):
		model.add(Dense(int({{uniform(num_classes,x_train.shape[1])}})))
		model.add(Dropout({{uniform(0,1)}}))
	if (num_layers in ['four','five']):
		model.add(Dense(int({{uniform(num_classes,x_train.shape[1])}})))
		model.add(Dropout({{uniform(0,1)}}))
	if (num_layers == 'five'):
		model.add(Dense(int({{uniform(num_classes,x_train.shape[1])}})))
		model.add(Dropout({{uniform(0,1)}}))

	# We have 6 classes, so output layer has 6
	model.add(Dense(num_classes, kernel_initializer='uniform', activation='softmax'))

	model.compile(loss='poisson', metrics=['accuracy'], optimizer='adam')
	model.fit(x_train, y_train, epochs=100, verbose=0, callbacks=[early_stop, reduce_LR])

	score, acc = model.evaluate(x_test, y_test, verbose=0)
	return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def metrics_report_to_df(ytrue, ypred):
	'''
	Given the set of true values and set of predicted values, returns  a table
	(dataframe) of the precision, recall, fscore, and support  for each class,
	including avg/total.
	'''
	# Analyze the predicted values
	precision, recall, fscore, support = metrics.precision_recall_fscore_support(ytrue, ypred, labels=mic_class_dict[drug])
	# Create a df for the report
	classification_report = pd.concat(map(pd.DataFrame, [precision, recall, fscore, support]), axis=1)
	classification_report.columns = ["precision", "recall", "f1-score", "support"]
	# Add a row for the average/total
	classification_report.loc['avg/Total', :] = metrics.precision_recall_fscore_support(ytrue, ypred, average='weighted')
	classification_report.loc['avg/Total', 'support'] = classification_report['support'].sum()
	return(classification_report)


if __name__ == "__main__":
	##################################################################
	# call with
	#	time python hyp.py <numfeats> <drug> <fold>
	# to do all folds
	#	for i in {1..5}; do python hyp.py <numfeats> <drug> '$i'; done
	# to do all folds on waffles
	#	sbatch -c 16 --mem 80G --partition NMLResearch --wrap='for i in {1..5}; do python hyp.py <numfeats> <drug> "$i"; done'
	# OR
	#   or use hyp.snake (change the features num)
	##################################################################

	feats = sys.argv[1]
	drug  = sys.argv[2]
	fold  = sys.argv[3]

	# Useful to have in the slurm output
	print("************************************")
	print("hyperas.py")
	print(drug, feats, fold)
	print("************************************")

	# Load data
	#filepath = os.path.abspath(os.path.curdir)+'/amr_data/'+drug+'/'+str(feats)+'feats/fold'+str(fold)+'/'
	mic_class_dict = joblib.load(os.path.abspath(os.path.curdir)+"/data/public_mic_class_order_dict.pkl")
	class_dict = mic_class_dict[drug]
	num_classes = len(mic_class_dict[drug])
	#genome_names = np.load(filepath+'genome_test.npy')

	# Split data, get best model
	train_data, train_names, test_data, test_names = data()
	best_run, best_model = optim.minimize(model=create_model, data=data, algo=tpe.suggest, max_evals=100, trials=Trials())

	# Find and record errors
	#find_errors(best_model, test_data, test_names, genome_names, class_dict, drug, mic_class_dict)

	## Score #######################################################
	score = best_model.evaluate(test_data, test_names)
	score_1d = eval_model(best_model, test_data, test_names)
	y_true = score_1d[3]
	y_true = y_true.astype(int)
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
	#best_model.save(filepath+'hyp_model.hdf5')
	#conf_df.to_pickle(filepath+'hyp_conf_df.pkl')
	#score_df.to_pickle(filepath+'hyp_score_df.pkl')
	#rep_df.to_pickle(filepath+'hyp_rep_df.pkl')
	with open('hyperas_out.txt','w') as f:
		f.write("\nBase acc: {0}%\n".format(round(Decimal(score[1]*100),2)))
		f.write("1-d acc: {0}%\n".format(score_1d[0]))
		f.write("MCC: {0}\n".format(round(score_1d[1],4)))
		f.write("\nConfusion Matrix\n{0}\n".format(conf_df))
		f.write("\nClassification Report\n{0}\n".format(report))
		f.write("Best performing model chosen hyper-parameters:\n{0}\n".format(best_run))
	################################################################
