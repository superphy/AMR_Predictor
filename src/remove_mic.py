import numpy as np
import os
import collections

drugs=["AMP","AMC","AZM","CHL","CIP","CRO","FIS","FOX","GEN","NAL","SXT","TET","TIO"]
datasets = ["", "grdi_", "kh_"]

#remove MIC classes that are less than 5
def remove_mic(X, Y):
	counts = collections.Counter(Y)
	counts = dict(counts)
	class_mask = np.array((len(Y)),dtype = 'object')
	low_freq = []
	for key, value in counts.items():
		if value < 5:
			low_freq = np.append(low_freq, key)
	class_mask = np.asarray([i not in low_freq for i in Y])

	return X[class_mask], Y[class_mask]

for data in datasets:
	for drug in drugs:
		 X = np.load(("data/{}{}/kmer_matrix.npy").format(data, drug))
		 Y = np.load(("data/{}{}/kmer_rows_mic.npy").format(data, drug))
		 X, Y = remove_mic(X, Y)

		 #create save location for new matrix/rows with removed MIC classes
		 if not os.path.exists(os.path.abspath(os.path.curdir)+'/data/filtered/'):
		 	os.mkdir(os.path.abspath(os.path.curdir)+'/data/filtered/{}{}').format(data, drug)
		 np.save('data/filtered/{}{}/kmer_matrix.npy', X).format(data, drug)
		 np.save('data/filtered/{}{}/kmer_rows_mic.npy', Y).format(data, drug)
