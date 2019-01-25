#!/usr/bin/env python


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from keras.utils import np_utils, to_categorical
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import matthews_corrcoef, classification_report, precision_recall_fscore_support
import collections
from sklearn.externals import joblib
import sys, os
from sklearn import preprocessing
import getopt
import pickle

from hpsklearn import HyperoptEstimator, svc, xgboost_classification
from hyperopt import tpe

from model_evaluators import *
from data_transformers import *

def get_data(dataset, drug):
	#choose which dataset to use
	path  = ""
	if dataset == "grdi":
		path = "grdi_"
	elif dataset == "kh":
		path = "kh_"
	else:
		if dataset != "public":
			raise Exception('did not receive a valid -x or -y name, run model.py --help for more info')

	#load kmer matrix and MIC classes
	X = np.load(("data/filtered/{}{}/kmer_matrix.npy").format(path,drug))
	Y = np.load(("data/filtered/{}{}/kmer_rows_mic.npy").format(path,drug))
	Z = np.load(("data/filtered/{}{}/kmer_rows_genomes.npy").format(path,drug))
	Y = [remove_symbols(i) for i in Y]
	return X, Y, Z

def usage():
	print("usage: model.py -x public -y grdi -f 3000 -a AMP\n\nOptions:")
	print(
	"-x, --train       Which set to train on [us, uk, uk_us, omnilog, kmer]",
	"-y, --test        Which set to test on [us, uk, uk_us, omnilog, kmer]",
	"                  Note that not passing a -y causes cross validation on the train set",
	"-f, --features    Number of features to train on, set to 0 to use all",
	"-a, --attribute   What to make the prediction on [Host, Serotype, Otype, Htype]",
	"-m, --model       Which model to use [XGB, SVM, ANN], defaults to XGB",
	"-o, --out         Where to save result DF, defaults to print to std out",
	"-p,               Add this flag to do hyperparameter optimization, XGB/SVM only",
	"-i,               Saves all features and their importance in data/features",
	"-e, 			   Saves errors to data/errors",
	"-h, --help        Prints this menu",
	sep = '\n')
	return

if __name__ == "__main__":
	"""
	#leave at 0 features for no feature selection
	num_feats = int(sys.argv[1])

	# can be Host, Serotype, Otype or Htype
	predict_for = sys.argv[2]

	# can be SVM, XGB, or ANN
	model_type = sys.argv[3]

	# can be kmer or omnilog
	source = sys.argv[4]
	"""
	train = ''
	test = 'cv'
	num_feats = 0
	predict_for = ''
	model_type = 'XGB'
	hyper_param = 0
	out = 'print'
	imp_feats = 0
	save_errors = 0

	OBO_acc = np.zeros((2,5))
	try:
		opts, args =  getopt.getopt(sys.argv[1:],"hx:y:f:a:m:o:pi",["help","train=","test=","features=","attribute=","model=","out="])
	except getopt.GetoptError:
		usage()
		sys.exit(2)
	for opt, arg, in opts:
		if opt in ('-x', '--train'):
			train = arg
		elif opt in ('-y', '--test'):
			test = arg
		elif opt in ('-f', '--features'):
			num_feats = int(arg)
		elif opt in ('-a', '--attribute'):
			predict_for = arg
		elif opt in ('-m', '--model'):
			model_type = arg
		elif opt in ('-o', '--out'):
			out = arg
		elif opt == '-p':
			hyper_param = 1
		elif opt == '-i':
			imp_feats = 1
			if not os.path.exists(os.path.abspath(os.path.curdir)+'/data/features'):
				os.mkdir(os.path.abspath(os.path.curdir)+'/data/features')
		elif opt == 'e':
			save_errors = 1
		elif opt in ('-h', '--help'):
			usage()
			sys.exit()

	X = []
	Y = []
	Z = []
	x_train = []
	x_test = []
	y_train = []
	y_test = []
	z_train = []
	z_test = []

	if(train=='grdi' and predict_for == 'FIS'):
		sys.exit()

	# this encodes the classes into integers for the models, 1,2,4,8 becomes 0,1,2,3
	le = preprocessing.LabelEncoder()
	if(train=='grdi'):
		mic_class_dict = joblib.load("data/grdi_mic_class_order_dict.pkl")
	else:
		mic_class_dict = joblib.load("data/public_mic_class_order_dict.pkl")

	#le.fit([remove_symbols(i) for i in mic_class_dict['predict_for']])

	mic_dict = [remove_symbols(i) for i in mic_class_dict[predict_for]]
	le.fit(mic_dict)
	#if no -y is passed in we want to do a 5 fold cross validation on the data
	if test =='cv':
		X, Y, Z = get_data(train, predict_for) # pull entire data set to be split later
		Y = le.transform(Y)
		#print(Y)
		#Y = encode_categories(Y, mic_dict)
		if(num_feats>= X.shape[1]):
			num_feats = 0
	else: #we are not cross validating
		x_train, y_train, z_train = get_data(train, predict_for) # pull training set
		x_test, y_test, z_test = get_data(test, predict_for) # pull testing set
		#le.fit(np.concatenate((y_train,y_test))) # need to fit on both sets of labels, so everything is accounted for (finding what the replacements are)
		y_train = le.transform(y_train) # just applying the label encoder on the 2 sets (actually replacing things)
		y_test = le.transform(y_test)
		#y_train = encode_categories(y_train, mic_dict)
		#y_test  = encode_categories(y_test, mic_dict)
		if(num_feats>= x_train.shape[1]):
			num_feats = 0

	#num_classes = len(le.classes_)
	num_classes = len(mic_class_dict[predict_for])

	num_threads = 8

	cv = StratifiedKFold(n_splits=5, random_state=913824)
	cvscores = []
	window_scores = []
	mcc_scores = []
	report_scores = []
	split_counter = 0

	train_string = train
	test_string = test

	model_data = [[train, test]]
	#if we are only using one set, we need to split it into multiple folds for training and testing
	if(test == 'cv'):
		model_data = cv.split(X,Y,Z)

	for train,test in model_data:
		split_counter +=1
		if(test_string=='cv'):
			x_train = X[train]
			x_test = X[test]
			y_test = Y[test]
			y_train = Y[train]

		cols = []
		#feature selection
		if(num_feats!=0):
			sk_obj = SelectKBest(f_classif, k=num_feats)
			x_train = sk_obj.fit_transform(x_train, y_train)
			x_test  = sk_obj.transform(x_test)
		"""
		#uncomment this section if you want to save datasets for hyp.property
		np.save('x_test.npy', x_test)
		np.save('x_train.npy', x_train)
		np.save('y_test.npy', y_test)
		np.save('y_train.npy', y_train)
		"""
		if(imp_feats):
			cols = np.load('data/unfiltered/kmer_cols.npy')
			feat_indices = np.zeros(len(cols))
			feat_indices = np.asarray([i for i in range(len(cols))])

			#creating an array of features that are persiting past feature selection
			feat_indices = sk_obj.transform(feat_indices.reshape(1,-1))
			feat_indices = feat_indices.flatten()
			top_feat_mask  = np.zeros(len(cols))
			top_feat_mask = np.asarray([i in feat_indices for i in range(len(cols))])
			cols = cols[top_feat_mask]

		if(model_type == 'XGB'):
			if(num_classes==2):
				objective = 'binary:logistic'
			else:
				objective = 'multi:softmax'
			if(hyper_param):
				model = HyperoptEstimator(classifier=xgboost_classification('xbc'), preprocessing=[], algo=tpe.suggest, trial_timeout=200)
			else:
				model = XGBClassifier(learning_rate=1, n_estimators=10, objective=objective, silent=True, nthread=num_threads)
			print(x_train, y_train)
			print(num_classes)
			model.fit(x_train,y_train)

			if(imp_feats):
				feat_save = 'data/features/'+predict_for+'_'+str(num_feats)+'feats_'+model_type+'trainedOn'+train_string+'_testedOn'+test_string+'_fold'+str(split_counter)+'.npy'
				np.save(feat_save, np.vstack((cols.flatten(), model.feature_importances_)))
			if(save_errors):
				find_errors(model, x_test, y_test, z_test, mic_class_dict[drug], drug, mic_class_dict, 'data/errors/'+predict_for+'_'+str(num_feats)+'feats_'+model_type+'trainedOn'+train_string+'_testedOn'+test_string+'_fold'+str(split_counter)+'.npy')

		elif(model_type == 'SVM'):
			from sklearn import svm
			if(hyper_param):
				model = HyperoptEstimator(classifier=svc("mySVC"), preprocessing=[], algo=tpe.suggest, trial_timeout=200)
			else:
				model = svm.SVC()
			model.fit(x_train,y_train)
			if(imp_feats):
				raise Exception('You can only pull feature importances from XGB, remove the -i or -m flags')
		elif(model_type == 'ANN'):
			if(hyper_param):
				raise Exception('This script does not support hyperas for ANN hyperparameter optimization, see src/hyp.py')
			if(imp_feats):
				raise Exception('You can only pull feature importances from XGB, remove the -i or -m flags')
			from keras.layers.core import Dense, Dropout, Activation
			from keras.models import Sequential
			from keras.utils import np_utils, to_categorical
			from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

			y_train = to_categorical(y_train, num_classes)
			y_test  = to_categorical(y_test, num_classes)

			patience = 16
			early_stop = EarlyStopping(monitor='loss', patience=patience, verbose=1, min_delta=0.005, mode='auto')
			model_save = ModelCheckpoint("best_model.hdf5",monitor='loss', verbose = 0, save_best_only =True, save_weights_only = False, mode ='auto', period =1)
			reduce_LR = ReduceLROnPlateau(monitor='loss', factor= 0.1, patience=(patience/2), verbose = 1, min_delta=0.005,mode = 'auto', cooldown=0, min_lr=0)

			model = Sequential()
			model.add(Dense(num_feats,activation='relu',input_dim=(num_feats)))
			model.add(Dropout(0.50))
			model.add(Dense(int(((num_feats+num_classes)/2)), activation='relu', kernel_initializer='uniform'))
			model.add(Dropout(0.50))
			model.add(Dense(num_classes, kernel_initializer='uniform', activation='softmax'))

			if(num_classes==2):
				loss = 'binary_crossentropy'
			else:
				loss = 'poisson'
			model.compile(loss=loss, metrics=['accuracy'], optimizer='adam')

			model.fit(x_train, y_train, epochs=100, verbose=1, callbacks=[early_stop, reduce_LR])
		else:
			raise Exception('Unrecognized Model. Use XGB, SVM or ANN')

		if(model_type == 'ANN'):
			results = ann_1d(model, x_test, y_test, 0)
			OBOResults = ann_1d(model, x_test, y_test, 1)
		else:
			results = xgb_tester(model, x_test, y_test, 0)
			OBOResults = xgb_tester(model, x_test, y_test, 1)
		print('OBO', OBOResults[0], len(y_test))
		print('OBN', results[0], len(y_test))
		OBO_acc[1, split_counter-1] = OBOResults[0]
		if(model_type == 'ANN'):
			OBO_acc[0, split_counter-1] = y_test.shape[0]
		else:
			OBO_acc[0, split_counter-1] = len(y_test)
		mcc_scores.append(results[1])

		labels = np.arange(0,num_classes)
		report = precision_recall_fscore_support(results[3], results[2], average=None, labels=labels)

		report_scores.append(report)
		cvscores.append(results[0])


	np.set_printoptions(suppress=True)
	avg_reports = np.mean(report_scores,axis=0)
	avg_reports = np.transpose(avg_reports)
	avg_reports = np.around(avg_reports, decimals=2)
	OBO_array = np.zeros((avg_reports.shape[0],1))
	OBO_sum = 0
	for i in range(5):
		OBO_sum += OBO_acc[1,i]/100 * OBO_acc[0,i]
	OBO_array[0,0] = OBO_sum/(np.sum(OBO_acc[0]))
	result_df = pd.DataFrame(data = np.hstack((avg_reports,OBO_array)), index = mic_class_dict[predict_for], columns = ['Precision','Recall', 'F-Score','Supports', '1D Acc'])
	running_sum = 0
	t_string = ''
	if(test_string == 'cv'):
		result_df.values[:,3] = [i*5 for i in result_df.values[:,3]]
		t_string = 'aCrossValidation'
		for row in result_df.values:
			running_sum+=(row[1]*row[3]/X.shape[0])
	else:
		t_string = test_string
		for row in result_df.values:
			running_sum+=(row[1]*row[3]/(len(y_test)))

	print("Predicting for", predict_for)
	print("on {} features using a {} trained on {} data, tested on {}".format(num_feats, model_type, train_string, t_string))
	print("Accuracy:", running_sum)
	print(result_df)
	if out != "print":
		if not (out.endswith('/')):
			out = out + '/'
		out = out+predict_for+'_'+str(num_feats)+'feats_'+model_type+'trainedOn'+train_string+'_testedOn'+t_string+'.pkl'
		result_df.to_pickle(out)
