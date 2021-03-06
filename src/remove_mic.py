import numpy as np
import os
import collections
import sys

def remove_mic(X, Y, Z):
	counts = collections.Counter(Y)
	counts = dict(counts)
	class_mask = np.array((len(Y)),dtype = 'object')
	low_freq = []
	for key, value in counts.items():
		if value < 5:
			low_freq = np.append(low_freq, key)
	class_mask = np.asarray([i not in low_freq for i in Y])

	return X[class_mask], Y[class_mask], Z[class_mask]
if __name__ == "__main__":
	drug = sys.argv[1]
	data = sys.argv[2]
	if data =='public':
		data = ""
	if not os.path.exists(os.path.abspath(os.path.curdir)+'/data/filtered/'):
	   os.mkdir(os.path.abspath(os.path.curdir)+'/data/filtered/')

	if(data=='grdi_' and drug =='FIS'):
		sys.exit()
	X = np.load(("data/{}{}/kmer_matrix.npy".format(data, drug)))
	Y = np.load(("data/{}{}/kmer_rows_mic.npy".format(data, drug)))
	Z = np.load(("data/{}{}/kmer_rows_genomes.npy".format(data, drug)))
	X, Y, Z = remove_mic(X, Y, Z)

	#create save location for new matrix/rows with removed MIC classes
	if not os.path.exists(os.path.abspath(os.path.curdir)+'/data/filtered/{}{}'.format(data, drug)):
		os.mkdir(os.path.abspath(os.path.curdir)+'/data/filtered/{}{}'.format(data, drug))
	np.save('data/filtered/{}{}/kmer_matrix.npy'.format(data, drug), X)
	np.save('data/filtered/{}{}/kmer_rows_mic.npy'.format(data, drug), Y)
	np.save('data/filtered/{}{}/kmer_rows_genomes.npy'.format(data, drug), Z)
