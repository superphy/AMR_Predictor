#!/usr/bin/env python

import numpy as np
import os
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif


def select(X, y, fmax, fmin, n):
	"""Apply a frequency and F-test feature selection to feature matrix. Return
	relevant columns that pass criteria.

	Args:
		X: kmer count feature matrix (rows: genomes, columns: kmers)
		y: MIC values
		fmax: float between 0-1. If kmer is found in more rows than floor(nrow*fmax), remove
		fmin: float between 0-1. If kmer is found in less rows than floor(nrow*fmax), remove
		n: number of columns to keep, after columns are sorted by F-score

	Returns:
		list of column indices

	"""

	# Mask too frequent and infrequent genomes
	if fmax < 1 or fmin > 0:
		mask = np.apply_along_axis(pass_freq_check, 1, X, fmax, fmin)
		X = X.loc[:, mask]

	# Mask genomes with F-test scores
	fsel = SelectKBest(f_classif, k=n)
	fsel.fit(X, y)
	idx = np.argwhere(np.logical_and(mask, fsel.get_support()))

	return idx[:,0]


def pass_freq_check(col, fmax, fmin):
	n = len(col)
	empty = np.count_nonzero(col==0)

	p = empty/n

	if p > fmax or p < fmin:
		return False
	else:
		return True


def dropna_and_encode_rows(y, ordarr):
	"""Remove NaN rows and change MIC text labels to ints. Labels are assiged
	order based on ordarr

	Args:
		y: MIC Series
		ordarr: Ordred list of MIC text labels (e.g. ["1.00", "2.00", ...])

	Returns:
		list of column indices

	"""

	# In pandas series objects, the row names are provided as index
	# So i don't need to keep track of which rows are removed, that info
	# is in index.

	if np.any(y == 'invalid'):
		raise Exception('I though all invalids were removed in the mics.snakefile')

	y = y.loc[~pd.isnull(y)]
	map_dict = { m: i  for i,m in enumerate(ordarr) }
	y = y.map(map_dict)

	if np.any(pd.isnull(y)):
		raise Exception('unexpected MIC value')

	return y


def apply():
	pass
