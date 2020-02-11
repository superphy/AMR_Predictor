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

from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=16, inter_op_parallelism_threads=16)))

from hyperas import optim
from hyperas.distributions import choice, uniform

from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import matthews_corrcoef, confusion_matrix, classification_report

from model_evaluators import *
from data_transformers import *

#seed(913824)
#set_random_seed(913824)

def explain_parameters(best_run):
	"""
	Takes in a hyperas best run dictionary and explains the network architecture
	"""
	print("--- model parameters ---")
	levels_of_patience = [4,8,12,16]
	print('Patience:',levels_of_patience[int(best_run['patience'])])
	num_layers = int(best_run['num_layers'])

	if num_layers == 0:
		print("no hidden layers")
	else:
		print('Input')
		for layer in range(num_layers):
			if layer == 0:
				print("{} neurons".format(best_run['int']))
				print("{}{} dropout".format(best_run['Dropout'],'%'))
			else:
				print("{} neurons".format(best_run['int_'+str(layer)]))
				print("{}{} dropout".format(best_run['Dropout_'+str(layer)],'%'))
		print('Output')

	print("--- end of parameters ---")

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
	from sklearn.feature_selection import SelectKBest, f_classif

	feats = int(sys.argv[1])
	drug  = sys.argv[2]
	fold  = sys.argv[4]
	dataset = sys.argv[5]
	kmer_length = sys.argv[6]

	# Matrix of classes for each drug
	if(dataset=='grdi'):
		mic_class_dict = joblib.load(os.path.abspath(os.path.curdir)+"/data/grdi_mic_class_order_dict.pkl")
	else:
		mic_class_dict = joblib.load(os.path.abspath(os.path.curdir)+"/data/public_mic_class_order_dict.pkl")

	path=''
	if(dataset=='grdi'):
		path = 'grdi_'
	elif(dataset=='kh'):
		path = 'kh_'

	if kmer_length != '11':
		kmer = '_'+kmer_length+'mer'
	else:
		kmer = ''

	# fold 1 uses sets 1,2,3 to train, 4 to test, fold 2 uses sets 2,3,4 to train, 5 to test, etc
	train_sets = [(i+int(fold)-1)%5 for i in range(5)]
	train_sets = [str(i+1) for i in train_sets]

	num_classes = len(mic_class_dict[drug])

	# load the relevant training sets and labels
	x_train1 = np.load('data/filtered/{}{}{}/splits/set{}/x.npy'.format(path,drug,kmer,train_sets[0]))
	x_train2 = np.load('data/filtered/{}{}{}/splits/set{}/x.npy'.format(path,drug,kmer,train_sets[1]))
	x_train3 = np.load('data/filtered/{}{}{}/splits/set{}/x.npy'.format(path,drug,kmer,train_sets[2]))
	y_train1 = np.load('data/filtered/{}{}{}/splits/set{}/y.npy'.format(path,drug,kmer,train_sets[0]))
	y_train2 = np.load('data/filtered/{}{}{}/splits/set{}/y.npy'.format(path,drug,kmer,train_sets[1]))
	y_train3 = np.load('data/filtered/{}{}{}/splits/set{}/y.npy'.format(path,drug,kmer,train_sets[2]))

	# merge the 3 training sets into 1
	x_train = np.vstack((x_train1, x_train2, x_train3))
	y_train = np.concatenate((y_train1, y_train2, y_train3))

	x_test  = np.load('data/filtered/{}{}{}/splits/set{}/x.npy'.format(path,drug,kmer,train_sets[3]))
	y_test  = np.load('data/filtered/{}{}{}/splits/set{}/y.npy'.format(path,drug,kmer,train_sets[3]))

	x_val  = np.load('data/filtered/{}{}{}/splits/set{}/x.npy'.format(path,drug,kmer,train_sets[4]))

	# hyperas asks for train and test so the validation set is what comes last, to check the final model
	# we need to save it to be used later, because we have the sk_obj now.
	if(feats!=0):
		sk_obj = SelectKBest(f_classif, k=feats)
		x_train = sk_obj.fit_transform(x_train, y_train)
		x_test  = sk_obj.transform(x_test)
		x_val  = sk_obj.transform(x_val)
		np.save('data/filtered/{}{}{}/splits/val{}_{}.npy'.format(path,drug,kmer,fold,str(feats)), x_val)

	y_train = to_categorical(y_train, num_classes)
	y_test  = to_categorical(y_test, num_classes)

	return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
	patience = {{choice([4,8,12,16])}}
	early_stop = EarlyStopping(monitor='loss', patience=patience, verbose=0, min_delta=0.005, mode='auto')
	model_save = ModelCheckpoint("best_model.hdf5",monitor='loss', verbose = 0, save_best_only =True, save_weights_only = False, mode ='auto', period =1)
	reduce_LR = ReduceLROnPlateau(monitor='loss', factor= 0.1, patience=(patience/2), verbose = 0, min_delta=0.005,mode = 'auto', cooldown=0, min_lr=0)

	model = Sequential()

	# how many hidden layers are in our model
	num_layers = {{choice(['zero', 'one', 'two', 'three', 'four', 'five'])}}

	if(num_layers == 'zero'):
		model.add(Dense(num_classes,activation='softmax',input_dim=(x_train.shape[1])))
	else:
		# this isnt a for loop because each variable needs its own name to be independently trained
		if (num_layers in ['one','two','three','four','five']):
			model.add(Dense(int({{uniform(num_classes,x_train.shape[1])}}),activation='relu',input_dim=(x_train.shape[1])))
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

		model.add(Dense(num_classes, kernel_initializer='uniform', activation='softmax'))

	model.compile(loss='poisson', metrics=['accuracy'], optimizer='adam')
	model.fit(x_train, y_train, epochs=100, verbose=0, batch_size=6000, callbacks=[early_stop, reduce_LR])

	score, acc = model.evaluate(x_test, y_test, verbose=0)
	return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def metrics_report_to_df(ytrue, ypred):
	'''
	DEPRECATED
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

	feats = sys.argv[1]
	drug  = sys.argv[2]
	max_evals = int(sys.argv[3])
	fold  = sys.argv[4]
	dataset = sys.argv[5]
	kmer_length = sys.argv[6]

	# Useful to have in the slurm output
	print("************************************")
	print("hyperas.py")
	print(drug, feats)
	print("************************************")
	print("Trials:",max_evals)

	# Load data
	mic_class_dict = joblib.load(os.path.abspath(os.path.curdir)+"/data/public_mic_class_order_dict.pkl")
	class_dict = mic_class_dict[drug]
	num_classes = len(mic_class_dict[drug])

	# Split data, get best model
	train_data, train_names, test_data, test_names = data()
	best_run, best_model = optim.minimize(model=create_model, data=data, algo=tpe.suggest, max_evals=max_evals, trials=Trials(),keep_temp=True)


	print("Validation accuracies")
	print(best_model.evaluate(test_data, test_names))
	print("ending validation acc")

	# Find and record errors
	# find_errors(best_model, test_data, test_names, genome_names, class_dict, drug, mic_class_dict)

	# load validation set
	path=''
	if(dataset=='grdi'):
		path = 'grdi_'
	elif(dataset=='kh'):
		path = 'kh_'

	if kmer_length != '11':
		kmer = '_'+kmer_length+'mer'
	else:
		kmer = ''

	# fold 1 uses sets 1,2,3 to train, 4 to test and 5 to validate, fold 2 uses sets 2,3,4 to train, 5 to test, etc
	train_sets = [(i+int(fold)-1)%5 for i in range(5)]
	train_sets = [str(i+1) for i in train_sets]

	test_names  = np.load('data/filtered/{}{}{}/splits/set{}/y.npy'.format(path,drug,kmer,train_sets[4]))
	test_data = np.load('data/filtered/{}{}{}/splits/val{}_{}.npy'.format(path,drug,kmer,fold,feats))
	from keras.utils import to_categorical
	test_names = to_categorical(test_names, num_classes)

	## Score #######################################################
	score = best_model.evaluate(test_data, test_names)
	explain_parameters(best_run)
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

	if not os.path.exists(os.path.abspath(os.path.curdir)+"/data/{}{}{}/hyperas/".format(path,drug,kmer)):
		os.makedirs(os.path.abspath(os.path.curdir)+"/data/{}{}{}/hyperas/".format(path,drug,kmer))

	# ann_1d -> returns: (perc, mcc, prediction, actual)
	results = ann_1d(best_model, test_data, test_names, 0)
	OBOResults = ann_1d(best_model, test_data, test_names, 1)

	labels = np.arange(0,num_classes)
	report = precision_recall_fscore_support(results[3], results[2], average=None, labels=labels)

	report_scores = []
	report_scores.append(report)

	# this is to match the result formatting of model.py to ensure XGB and ANN are treated the same
	np.set_printoptions(suppress=True)
	avg_reports = np.mean(report_scores, axis=0)
	avg_reports = np.transpose(avg_reports)
	avg_reports = np.around(avg_reports, decimals=2)
	OBO_array = np.zeros((avg_reports.shape[0],1))
	OBO_array[0,0] = OBOResults[0]/100

	result_df = pd.DataFrame(data = np.hstack((avg_reports,OBO_array)), index = mic_class_dict[drug], columns = ['Precision','Recall', 'F-Score','Supports', '1D Acc'])

	running_sum = 0
	t_string = 'aCrossValidation'
	for row in result_df.values:
		running_sum+=(row[1]*row[3]/(len(test_names)))

	print("Predicting for", drug)
	print("on {} features using a {} trained on {} data, tested on {}".format(feats, 'ANN', 'public', t_string))
	print("Accuracy:", running_sum)
	print(result_df)
	out = "data/{}{}{}/hyperas/".format(path, drug, kmer)
	out = out+str(feats)+'feats_'+fold+'.pkl'
	result_df.to_pickle(out)
