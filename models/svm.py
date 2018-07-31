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


seed(913824)
set_random_seed(913824)


def eval_model(model, test_data, test_names):
	'''
	Takes a model (neural net), a set of test data, and a set of test names.
	Returns perc: the precent of correct guesses by the model within 1 dilution.
	Returns mcc: the matthews correlation coefficient.
	Returns prediction and actual.
	'''
	# Create the prediction from the model
	prediction = model.predict(test_data)
	prediction = [int(round(float(value))) for value in prediction]

	actual = test_names
	actual = [int(float(value)) for value in actual]
	# Sum the number of correct guesses within 1 dilution: if the bin is one to either
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

	# Find the matthew's coefficient
	mcc = matthews_corrcoef(np.argmax(to_categorical(actual),axis=1),(prediction))
	return (perc, mcc, prediction, actual)

def eval_modelOBO(model, test_data, test_names):
	'''
	Takes a model (neural net), a set of test data, and a set of test names.
	Returns perc: the precent of correct guesses by the model within 1 dilution.
	Returns mcc: the matthews correlation coefficient.
	Returns prediction and actual.
	'''
	# Create the prediction from the model
	prediction = model.predict(test_data)
	prediction = [int(round(float(value))) for value in prediction]

	actual = test_names
	actual = [int(float(value)) for value in actual]
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
	#mcc = 1
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
	if not os.path.exists(os.path.abspath(os.path.curdir)+'/amr_data/errors'):
		os.mkdir(os.path.abspath(os.path.curdir)+'/amr_data/errors')
	err_file = open(os.path.abspath(os.path.curdir)+'/amr_data/errors/'+str(sys.argv[1])+'_feats_svm_errors.txt', 'a+')

	prediction = model.predict(test_data)
	prediction = [int(round(float(value))) for value in prediction]
	actual = [int(float(value)) for value in test_names]

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


def metrics_report_to_df(ytrue, ypred):
	precision, recall, fscore, support = metrics.precision_recall_fscore_support(ytrue, ypred, labels=mic_class_dict[drug])
	classification_report = pd.concat(map(pd.DataFrame, [precision, recall, fscore, support]), axis=1)
	#classification_report.set_axis(mic_class_dict[drug], axis='index', inplace=True)
	classification_report.columns = ["precision", "recall", "f1-score", "support"] # Add row w "avg/total"
	classification_report.loc['avg/Total', :] = metrics.precision_recall_fscore_support(ytrue, ypred, average='weighted')
	classification_report.loc['avg/Total', 'support'] = classification_report['support'].sum()
	return(classification_report)


if __name__ == "__main__":
	##################################################################
	# call with
	#	time python svm.py <numfeats> <drug> <fold>
	# to do all folds
	#	for i in {1..5}; do python svm.py <numfeats> <drug> '$i'; done
	# to do all folds on waffles
	#	sbatch -c 16 --mem 80G --partition NMLResearch --wrap='for i in {1..5}; do python svm.py <numfeats> <drug> "$i"; done'
	# OR
	#   or use svm.snake (change the features num)
	##################################################################

	feats = sys.argv[1]
	drug = sys.argv[2]
	fold = sys.argv[3]

	# Useful to have in the slurm output
	print("************************************")
	print("xgboost.py")
	print(drug, feats, fold)
	print("************************************")

	# Load data
	mic_class_dict = joblib.load(os.path.abspath(os.path.curdir)+"/amr_data/mic_class_order_dict.pkl")
	class_dict = mic_class_dict[drug]
	num_classes = len(mic_class_dict[drug])
	filepath = os.path.abspath(os.path.curdir)+'/amr_data/'+drug+'/'+str(feats)+'feats/fold'+str(fold)+'/'
	genome_names = np.load(filepath+'genome_test.npy')

	# Load training and testing sets
	x_train = np.load(filepath+'x_train.npy')
	x_test  = np.load(filepath+'x_test.npy')
	y_train = np.load(filepath+'y_train.npy')
	y_test  = np.load(filepath+'y_test.npy')

	model = HyperoptEstimator(classifier=svc("mySVC"), preprocessing=[], algo=tpe.suggest, max_evals=100, trial_timeout=120)
	model.fit(x_train, y_train)
	best_model = model.best_model()

	# Find and record errors
	find_errors(model, x_test, y_test, genome_names, class_dict, drug, mic_class_dict)

	## Score #######################################################
	score = eval_model(model, x_test, y_test)
	score_1d = eval_modelOBO(model, x_test, y_test)
	y_true = score[3]
	y_pred = score[2]
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
	pickle.dump(model, open(filepath+'svm_model.dat', 'wb'))
	conf_df.to_pickle(filepath+'svm_conf_df.pkl')
	score_df.to_pickle(filepath+'svm_score_df.pkl')
	rep_df.to_pickle(filepath+'svm_rep_df.pkl')
	with open(filepath+'svm_out.txt','w') as f:
		f.write("\nBase acc: {0}%\n".format(score[0]))
		f.write("1-d acc: {0}%\n".format(score_1d[0]))
		f.write("MCC: {0}\n".format(round(score_1d[1],4)))
		f.write("\nConfusion Matrix\n{0}\n".format(conf_df))		
		f.write("\nClassification Report\n{0}\n".format(report))
		f.write("Best performing model chosen hyper-parameters:\n{0}".format(best_model))
	################################################################
