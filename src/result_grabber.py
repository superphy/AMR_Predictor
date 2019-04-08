"""
This is a script to return the results of a specific test
"""
import pandas as pd
import numpy as np
import sys, getopt
import glob

import warnings
warnings.filterwarnings('ignore')

def usage():
	print("usage: result_grabber.py -x public -y grdi -f 3000 -a AMP -m ANN\n\nOptions:")
	print(
	"-x, --train       Which set to train on [public, grdi, kh]",
	"-y, --test        Which set to test on [public, grdi, kh]",
	"                  Note that not passing a -y causes cross validation on the train set",
	"-f, --features    Number of features to train on",
	"-a, --attribute   What to make the prediction on [AMP, AMC, AZM, etc]",
	"-m, --model       Which model to use [XGB, SVM, ANN], defaults to XGB",
	"-h, --help        Prints this menu",
	sep = '\n')
	return

if __name__ == "__main__":
	train = ''
	test = 'aCrossValidation'
	num_feats = 0
	predict_for = ''
	model_type = 'XGB'

	try:
		opts, args =  getopt.getopt(sys.argv[1:],"hx:y:f:a:m:",["help","train=","test=","features=","attribute=","model="])
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
		elif opt in ('-h', '--help'):
			usage()
			sys.exit(2)

	out = 'results/*/'+predict_for+'_'+str(num_feats)+'feats_'+model_type+'trainedOn'+train+'_testedOn'+test+'.pkl'
	for filename in glob.glob(out):
		result_df = pd.read_pickle(filename)
		num_samples = np.sum(result_df.values[:,3])
		#print(num_samples)
		correct = 0
		for row in result_df.values:
			correct += row[1] * row[3]

		print("Accuracy:", correct/num_samples)
		print("1D-Acc  :", result_df.values[0,4])
