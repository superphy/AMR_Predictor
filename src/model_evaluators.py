#!/usr/bin/env python

import numpy as np
from decimal import Decimal
import collections
from sklearn.metrics import matthews_corrcoef, classification_report, precision_recall_fscore_support
from keras.utils import np_utils, to_categorical

def ann_1d(model, test_data, test_names, dilution):
	'''
	Takes a model (neural net), a set of test data, and a set of test names.
	Returns perc: the percent of correct guesses by the model using a window of size 1.
	Returns mcc: the matthews correlation coefficient.
	Returns prediction and actual.
	'''
	# Create and save the prediction from the model
	prediction = model.predict_classes(test_data)
	#np.save('prediction.npy', prediction)

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
		if abs(pred-act)<=dilution :
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

def xgb_tester(model, test_data, test_names, dilution):
	'''
	Takes a model (xgboost), a set of test data, a set of test names and a dilution accuracy.
	Returns perc: the percent of correct guesses
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
		if abs(pred-act)<=dilution :
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
			print("Drug:{} Genome:{} Predicted:{} Actual:{} OBO:{} Major?:{}".format(drug, genome_names[i], class_dict[pred], class_dict[int(act)], off_by_one,find_major(pred,act,drug,mic_class_dict)))

	print("{} out of {} were incorrect ({} were close)".format(wrong_count, total_count, close_count))
