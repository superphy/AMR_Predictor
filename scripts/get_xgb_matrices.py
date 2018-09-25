import numpy as np
from sklearn.externals import joblib
import os


df = joblib.load(os.path.abspath(os.path.curdir)+"/amr_data/mic_class_dataframe.pkl")
df_rows = df.index.values 	# Row names are genomes
df_cols = df.columns

# For each drug
for drug in df_cols:

	filepath = os.path.abspath(os.path.curdir)+'/amr_data/'+drug+'/270feats/'

	fold1 = np.load(filepath+"fold1/xgb_prob_matrix.npy")
	fold2 = np.load(filepath+"fold2/xgb_prob_matrix.npy")
	fold3 = np.load(filepath+"fold3/xgb_prob_matrix.npy")
	fold4 = np.load(filepath+"fold4/xgb_prob_matrix.npy")
	fold5 = np.load(filepath+"fold5/xgb_prob_matrix.npy")

	matrix = np.concatenate((fold1, fold2, fold3, fold4, fold5), axis=0)

	np.save("xgb_matrices/"+drug+".npy", matrix)