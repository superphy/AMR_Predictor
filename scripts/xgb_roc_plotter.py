import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import os


def get_data(x, guess):

	print(x.shape)

	# Get unique sorted probability values
	probs = pd.DataFrame({'p': x, 'c': guess})
	probs = probs.groupby(['p']).agg(['sum','count'])
	probs = probs.sort_index(ascending=False)
	print(probs)

	# Compute accuracy for each cutoff
	# This sums total correct and total predictions using cumulative sums from highest to lowest probability values
	acc = pd.DataFrame({'cs': np.cumsum(probs['c']['sum']), 'cc': np.cumsum(probs['c']['count']) })
	acc['acc'] = acc['cs']/acc['cc']
	print(acc)

	return(acc)

def get_npreds(x, guess):

	print(x.shape)

	# Get unique sorted probability values
	probs = pd.DataFrame({'p': x, 'c': guess})
	probs = probs.groupby(['p']).agg(['sum','count'])
	probs = probs.sort_index(ascending=False)
	print(probs)

	# Compute accuracy for each cutoff
	# This sums total correct and total predictions using cumulative sums from highest to lowest probability values
	acc = pd.DataFrame({'cs': np.cumsum(probs['c']['sum']), 'cc': np.cumsum(probs['c']['count']) })
	tot=np.sum(probs['c']['count'])
	npreds = acc['cc']/tot
	acc['npreds'] = npreds

	return(acc)



if __name__ == "__main__":

	df = joblib.load(os.path.abspath(os.path.curdir)+"/amr_data/mic_class_dataframe.pkl")
	df_rows = df.index.values 	# Row names are genomes
	df_cols = df.columns

	f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12), (ax13, ax14, ax15)) = plt.subplots(5, 3)
	ax_res = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15]

	g, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12), (ax13, ax14, ax15)) = plt.subplots(5, 3)
	ax_res2 = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15]

	# For each drug
	counter = 0
	for drug in df_cols:
		f1_ax = ax_res[counter]
		f2_ax = ax_res2[counter]

		matrix = np.load("xgb_matrices/"+drug+".npy")

		first  = matrix[:,0] # first col
		print(first)
		second = matrix[:,1] # second col
		third  = matrix[:,2] # third col
		guess  = matrix[:,3] # 4th col
		first_guess  = (np.zeros((1,matrix.shape[0]))).flatten()
		second_guess = (np.zeros((1,matrix.shape[0]))).flatten()
		third_guess  = (np.zeros((1,matrix.shape[0]))).flatten()

		second = np.asarray([first[i]+second[i] for i in range(len(first))])
		third  = np.asarray([third[i]+second[i] for i in range(len(third))])
		
		# determining if the guess is in the range that we find correct
		for i in range(len(guess)):
			if guess[i] == 1:
				first_guess[i] = 1
				second_guess[i] = 1
				third_guess[i] = 1
			elif guess[i] == 2:
				first_guess[i] = 0
				second_guess[i] = 1
				third_guess[i] = 1
			elif guess[i] == 3:
				first_guess[i] = 0
				second_guess[i] = 0
				third_guess[i] = 1
			else:
				first_guess[i] = 0
				second_guess[i] = 0
				third_guess[i] = 0

		set1 = get_data(first, first_guess)
		set2 = get_data(second, second_guess)
		set3 = get_data(third, third_guess)

		f1_ax.plot(set1.index, set1['acc'])
		f1_ax.plot(set2.index, set2['acc'])
		f1_ax.plot(set3.index, set3['acc'])
		f1_ax.set_title(drug)

		set1 = get_npreds(first, first_guess)
		set2 = get_npreds(second, second_guess)
		set3 = get_npreds(third, third_guess)

		f2_ax.plot(set1.index, set1['npreds'])
		f2_ax.plot(set2.index, set2['npreds'])
		f2_ax.plot(set3.index, set3['npreds'])
		f2_ax.set_title(drug)

		counter+=1

	st = f.suptitle("xgboost - acc", fontsize=20)
	st = g.suptitle("xgboost - npreds", fontsize=20)
	plt.show()

