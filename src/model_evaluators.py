#!/usr/bin/env python

import numpy as np
from decimal import Decimal
import collections
from sklearn.metrics import matthews_corrcoef, classification_report, precision_recall_fscore_support
from keras.utils import np_utils, to_categorical

def ann_1d(model, test_data, test_names):
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
        if pred==act or pred==act+1 or pred==act-1:
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
