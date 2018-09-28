#!/usr/bin/env python

import numpy as np
import sys
from copy import deepcopy
import os

def find_index(kmer):
	for i, col in enumerate(heatmap_cols):
		if(kmer == col):
			return i

def collapse(heatmap, heatmap_cols, set_type):
	collapsed_matrix = np.empty((heatmap.shape[1],heatmap.shape[1]), dtype='object')
	collapsed_matrix[:] = 'X'

	to_be_sorted = deepcopy(heatmap_cols)

	for indx, col in enumerate(heatmap_cols):
		if col in to_be_sorted:
			next_open_row = 0
			for i in range(len(heatmap_cols)):
				if collapsed_matrix[i][0]=='X':
					next_open_row = i
					break
			collapsed_matrix[next_open_row][0] = col
			to_be_sorted[indx] = 'EMPTY'

			#go through heatmap for the row in find index, then append anything < diff_cuttoff and remove it from to be sorted
			duplicate_counter = 1
			for col_index, element in enumerate(heatmap[indx]):
				if(element <= diff_cutoff):
					target_kmer = heatmap_cols[col_index]
					if target_kmer in to_be_sorted:
						collapsed_matrix[next_open_row][duplicate_counter] = target_kmer
						to_be_sorted[col_index] = 'EMPTY'
	np.save('no_dup_feats/'+drug+'_'+set_type+'_collapsed_matrix.npy', collapsed_matrix)
	return collapsed_matrix

if __name__ == "__main__":
	drug = sys.argv[1]
	diff_cutoff = 5

	heatmap = np.load('no_dup_feats/'+drug+'_heatmap_matrix.npy')
	heatmap_cols = np.load('no_dup_feats/'+drug+'_heatmap_kmer_cols.npy')
	original_collapsed_matrix = collapse(heatmap, heatmap_cols, 'orig')

	grdi_heatmap = np.load('no_dup_feats/grdi_heatmaps/'+drug+'_heatmap_matrix.npy')
	grdi_heatmap_cols = np.load('no_dup_feats/grdi_heatmaps/'+drug+'_heatmap_kmer_cols.npy')
	grdi_collapsed_matrix = collapse(grdi_heatmap, grdi_heatmap_cols, 'grdi')


	kmer_matrix = np.load((os.path.abspath(os.path.curdir)+'/non_grdi/amr_data/'+drug+'/kmer_matrix.npy'))
	kmer_cols   = np.load((os.path.abspath(os.path.curdir)+'/non_grdi/amr_data/'+drug+'/kmer_cols.npy'))

	grdi_kmer_matrix = np.load((os.path.abspath(os.path.curdir)+'/amr_data/'+drug+'/kmer_matrix.npy'))
	grdi_kmer_cols   = np.load((os.path.abspath(os.path.curdir)+'/amr_data/'+drug+'/kmer_cols.npy'))

	assert(kmer_matrix.shape[1] == grdi_kmer_matrix.shape[1]), "Matrices do not save the same number of features"
	assert(len(kmer_cols) == len(grdi_kmer_cols)), "Matrices labels are of unequal length"
	assert(len(kmer_cols) == kmer_matrix.shape[1]), "Matrices have a different number of features as the feature labels"

	copy_mask = np.zeros(kmer_matrix.shape[1])
	for i in range(kmer_matrix.shape[1]):
		if(kmer_cols[i] in original_collapsed_matrix[:,0] or kmer_cols[i] in grdi_collapsed_matrix[:,0]):
			copy_mask[i] = 1
	copy_mask = [i==1 for i in copy_mask]

	print("***Before Mask***")
	print("kmer_matrix", kmer_matrix.shape)
	print("kmer_cols", len(kmer_cols))
	print("grdi_kmer_matrix", grdi_kmer_matrix.shape)
	print("grdi_kmer_cols", len(grdi_kmer_cols))

	kmer_matrix = kmer_matrix[:, copy_mask]
	kmer_cols = kmer_cols[copy_mask]
	grdi_kmer_matrix = grdi_kmer_matrix[:, copy_mask]
	grdi_kmer_cols = grdi_kmer_cols[copy_mask]

	print("***After Mask***")
	print("kmer_matrix", kmer_matrix.shape)
	print("kmer_cols", len(kmer_cols))
	print("grdi_kmer_matrix", grdi_kmer_matrix.shape)
	print("grdi_kmer_cols", len(grdi_kmer_cols))

	np.save((os.path.abspath(os.path.curdir))+'/no_dup_feats/'+str(diff_cutoff)+'diff/'+drug+'_kmer_matrix.npy', kmer_matrix)
	np.save((os.path.abspath(os.path.curdir))+'/no_dup_feats/'+str(diff_cutoff)+'diff/'+drug+'_kmer_cols.npy', kmer_cols)
	np.save((os.path.abspath(os.path.curdir))+'/no_dup_feats/'+str(diff_cutoff)+'diff/'+drug+'_grdi_kmer_matrix.npy', grdi_kmer_matrix)
	np.save((os.path.abspath(os.path.curdir))+'/no_dup_feats/'+str(diff_cutoff)+'diff/'+drug+'_grdi_kmer_cols.npy', grdi_kmer_cols)
