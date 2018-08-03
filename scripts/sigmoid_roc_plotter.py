import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import os

def get_data(x, guess):
	acc_matrix = np.zeros((100,2260))
	counter = 0
	print(x.shape)
	for j in range(len(x)):
		prob =int((x[j]*100))-1
		if(prob > 99):
			prob = 99
		if guess[j] == 0:
			acc_matrix[prob ,counter] = 2
		else:
			acc_matrix[prob ,counter] = 1
		counter +=1

	x_axis_acc = (np.zeros((1,100))).flatten()
	for k in range(100):
		correct_count = np.count_nonzero(acc_matrix[k,:]==1) #count number of 1 that are in acc_matrix row k
		wrong_count   = np.count_nonzero(acc_matrix[k,:]==2) #count number of 2 that are in acc_matrix row k
		if(correct_count!=0 and wrong_count!=0):
			x_axis_acc[k] = correct_count/(correct_count+wrong_count)
		else:
			x_axis_acc[k] = 0

	return x_axis_acc


if __name__ == "__main__":

	df = joblib.load(os.path.abspath(os.path.curdir)+"/amr_data/mic_class_dataframe.pkl")
	df_rows = df.index.values 	# Row names are genomes

	# find all drug names
	df_cols = df.columns

	# prepare plot area
	f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12), (ax13, ax14, ax15)) = plt.subplots(5, 3)
	ax_res = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15]

	# For each drug
	counter = 0
	for drug in df_cols:
		ax = ax_res[counter]
		'''
		print("start: prepping amr data for ",drug)

		filepath = os.path.abspath(os.path.curdir)+'/amr_data/'+drug+'/270feats/'

		fold1 = np.load(filepath+"fold1/sigmoid_prob_matrix.npy")
		fold2 = np.load(filepath+"fold2/sigmoid_prob_matrix.npy")
		fold3 = np.load(filepath+"fold3/sigmoid_prob_matrix.npy")
		fold4 = np.load(filepath+"fold4/sigmoid_prob_matrix.npy")
		fold5 = np.load(filepath+"fold5/sigmoid_prob_matrix.npy")

		matrix = np.concatenate((fold1, fold2, fold3, fold4, fold5), axis=0)
		'''

		matrix = np.load("sigmoid_matrices/"+drug+".npy")
		# matrix is a 2d array with 4 columns, they are:
		# Col 1: Sigmoid Probability of being in the 1st choice
		# Col 2: Sigmoid Probability of being in the 2nd choice
		# Col 3: Sigmoid Probability of being in the 3rd choice
		# Col 4: What choice was actually correct

		first  = matrix[:,0] # first col
		second = matrix[:,1] # 2nd col

		# adding the probability of the 1st and 2nd prediction
		second = np.asarray([first[i]+second[i] for i in range(len(first))])
		third  = matrix[:,2] # 3rd col

		# adding the probability of the 1st, 2nd, and 3rd prediction
		third  = np.asarray([third[i]+second[i] for i in range(len(third))])
		guess  = matrix[:,3] # 4th col

		print(first.shape, second.shape, third.shape)

		first_guess  = (np.zeros((1,2260))).flatten()
		second_guess = (np.zeros((1,2260))).flatten()
		third_guess  = (np.zeros((1,2260))).flatten()

		print(first[1], second[1], third[1])


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


		#fig = plt.figure()
		sigmoid_prediction_index = (range(100))
		ax.plot([set1],[sigmoid_prediction_index[1:]], 'r.')
		ax.plot([set2],[sigmoid_prediction_index[1:]], 'b.')
		ax.plot([set3],[sigmoid_prediction_index[1:]], 'g*')
		#ax.title(drug)
		ax.set_title(drug)
		#z = np.polyfit(set1,sigmoid_prediction_index,1)
		#plt.plot(z)
		#plt.title(drug+' sigmoid')
		counter+=1
	st = f.suptitle("Sigmoid", fontsize=20)
	plt.show()



Collapse 
