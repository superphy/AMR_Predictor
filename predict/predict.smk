

""" PIPELINE
clean
count - Jellyfish
dump - jellyfish
predict - load into matrix, spit out csv(from pandas)
evaluate - if they have MIC's listed, evaluate the model
"""
ids, = glob_wildcards("predict/genomes/raw/{id}.fasta")
features = 1000

def make_row(filename):
	from Bio import Seq, SeqIO
	import numpy as np
	"""
	Given a genome file, create and return a row of kmer counts
	to be inserted into the kmer matrix.

	DEPRECATED

	"""
	relevant_feats = np.load("predict/features/relevant_feats_{}.npy".format(str(features)))
	cols_dict = { relevant_feats[i] : i for i in range(0, len(relevant_feats))}

	# Create a temp row to fill and return (later placed in the kmer_matrix)
	temp_row = [0]*len(relevant_feats)

	# Walk through the file
	for record in SeqIO.parse("predict/genomes/jellyfish_results/"+filename, "fasta"):
		# Retrieve the sequence as a string
		kmer_seq = record.seq
		kmer_seq = kmer_seq._get_seq_str_and_check_alphabet(kmer_seq)


		if(kmer_seq in relevant_feats):
			kmer_count = int(record.id)
			temp_row[cols_dict[kmer_seq]] = kmer_count

	return filename, temp_row

def make_row_from_query(filename):
	import numpy as np
	relevant_feats = np.load("predict/features/relevant_feats_{}.npy".format(str(features)))
	cols_dict = { relevant_feats[i] : i for i in range(0, len(relevant_feats))}

	# Create a temp row to fill and return (later placed in the kmer_matrix)
	temp_row = [0]*len(relevant_feats)

	# place kmer counts in correct order
	with open("predict/genomes/jellyfish_results/"+filename) as file:
		for line in file:
			line = line.rstrip()
			kmer, count = line.split()
			if(int(count)>255):
				count = 255
			temp_row[cols_dict[kmer]] = int(count)

	# returns a row of kmer counts and the sequence name it came from
	return filename, temp_row

rule all:
	input:
		"predict/predictions.csv"

rule clean:
	# This rule cleans fastas that have low coverage or have sequencing/assembly errors
	input:
		"predict/genomes/raw/{id}.fasta"
	output:
		"predict/genomes/clean/{id}.fasta"
	shell:
		"python src/clean.py {input} predict/genomes/clean/"

rule count:
	# Rules count and dump count the number of appearences of each kmer of length 11
	input:
		"predict/genomes/clean/{id}.fasta"
	output:
		temp("predict/genomes/jellyfish_results/{id}.jf")
	threads:
		2
	shell:
		"jellyfish count -C -m 11 -s 100M -t {threads} {input} -o {output}"

rule dump:
	input:
		"predict/genomes/jellyfish_results/{id}.jf"
	output:
		"predict/genomes/jellyfish_results/{id}.fa"
	shell:
		"jellyfish query {input} -s predict/features/relevant_feats_1000.fasta > {output}"

rule matrix:
	# This rule is preprocessing the data to be in the correct format to be fed
	# into the machine learning models
	input:
		expand("predict/genomes/jellyfish_results/{id}.fa", id=ids)
	output:
		"predict/genomes/unfiltered/kmer_matrix.npy",
		"predict/genomes/unfiltered/kmer_rows.npy",
		"predict/genomes/unfiltered/kmer_cols.npy"
	threads:
		144
	run:
		import os, sys
		import numpy as np
		from concurrent.futures import ProcessPoolExecutor
		from multiprocessing import cpu_count

		num_start = 0
		num_stop = 0
		total = 0

		def progress():
			sys.stdout.write('\r')
			sys.stdout.write("Loading Genomes: {} started, {} finished, {} total".format(num_start,num_stop,total))
			sys.stdout.flush()
			if(num_stop==total):
				print("\nAll Genomes Loaded!\n")

		# find all the possible features that are going to be used to make the prediction
		if not os.path.exists(os.path.abspath(os.path.curdir)+"/predict/features/relevant_feats_{}.npy".format(str(features))):
			relevant_feats = []
			for feat_array in ([files for r,d,files in os.walk("predict/features/")][0]):
				relevant_feats = np.concatenate((relevant_feats, np.load("predict/features/"+feat_array)))

			# remove any duplicates
			relevant_feats = [i.decode('utf-8') for i in set(relevant_feats)]
			np.save("predict/features/relevant_feats_{}.npy".format(str(features)),relevant_feats)
		else:
			relevant_feats = np.load("predict/features/relevant_feats_{}.npy".format(str(features)))

		# find all the genomes we were given, genomes are filenames and runs are sample names
		genomes = ([files for r,d,files in os.walk("predict/genomes/jellyfish_results/")][0])
		total = len(genomes)
		runs = [i.split('.')[0] for i in genomes]

		# declaring empty kmer matrix to fill
		kmer_matrix = np.zeros((len(genomes),len(relevant_feats)),dtype = 'uint8')

		# making dicts for faster indexing
		# note that rows dict is in filenames not genome/run names
		rows_dict = { genomes[i] : i for i in range(0, len(genomes))}
		cols_dict = { relevant_feats[i] : i for i in range(0, len(relevant_feats))}

		# Use concurrent futures to get multiple rows at the same time
		# Then place completed rows into the matrix and update the row dictionary
		num_start += min(cpu_count(),len(genomes))
		progress()
		with ProcessPoolExecutor(max_workers=cpu_count()) as ppe:
			for genome_name,temp_row in ppe.map(make_row_from_query, genomes):
				num_stop+=1
				if(num_start<total):
					num_start+=1
				progress()
				for i, val in enumerate(temp_row):
					kmer_matrix[rows_dict[genome_name]][i] = val

		# save everything
		np.save("predict/genomes/unfiltered/kmer_matrix.npy", kmer_matrix)
		np.save("predict/genomes/unfiltered/kmer_rows.npy", runs)
		np.save("predict/genomes/unfiltered/kmer_cols.npy", relevant_feats)



rule predict:
	# This rule feeds the given genomes through the model and returns a .csv of the predicted MIC values
	input:
		"predict/genomes/unfiltered/kmer_matrix.npy",
		"predict/genomes/unfiltered/kmer_rows.npy",
		"predict/genomes/unfiltered/kmer_cols.npy"
	output:
		"predict/predictions.csv"
	run:
		import numpy as np
		import pandas as pd
		import pickle
		from sklearn import preprocessing
		from sklearn.externals import joblib
		import os,sys
		import xgboost as xgb
		from collections import Counter

		sys.path.insert(0, os.path.abspath(os.path.curdir)+"/src/")
		from data_transformers import remove_symbols
		from model_evaluators import find_major

		#drugs = ['AMP','AMC','AZM','CHL','CIP','CRO','FIS','FOX','GEN','NAL','SXT','TET','TIO']

		mic_class_dict = joblib.load("predict/genomes/public_mic_class_order_dict.pkl")
		drugs = list(mic_class_dict.keys())

		# load in 2D matrix of kmer counts, columns are kmers and rows are genomes so
		# kmer_matrix[genome][kmer] returns how many times that kmer was seen in that genome
		kmer_matrix = np.load("predict/genomes/unfiltered/kmer_matrix.npy")

		# load in labels for the predictions, if supplied
		evaluate = False
		try:
			labels = pd.read_excel("predict/mic_labels.xlsx")
			mic_df = joblib.load("predict/genomes/public_mic_class_dataframe.pkl")
			evaluate = True
		except:
			print("No valid MIC labels provided, will return predictions without evaluation")

		# load labels from last ruls
		kmer_rows = np.load("predict/genomes/unfiltered/kmer_rows.npy")
		kmer_cols = np.load("predict/genomes/unfiltered/kmer_cols.npy")

		# make pandas to store predictions
		predicts = np.zeros((len(kmer_rows), len(drugs)),dtype= 'object')
		predict_df = pd.DataFrame(data = predicts, index = kmer_rows, columns = drugs)

		# go through each drug and make prediction for just that drug
		for drug in drugs:
			# just use the data relevant to this drug, drug feats is in bytes, kmer_cols is not
			drug_feats = np.load("predict/features/"+str(features)+"feats_"+drug+".npy")
			drug_feats = [i.decode('utf-8') for i in drug_feats]

			cols_dict = { drug_feats[i] : i for i in range(0, len(drug_feats))}
			feat_mask = [i in drug_feats for i in kmer_cols]

			# transpose, apply mask, transpose again
			curr_matrix = np.transpose(np.transpose(kmer_matrix)[feat_mask])
			curr_cols = kmer_cols[feat_mask]

			# now we have the correct columns, but they are most likely in the wrong order
			# for every kmer in current order, find new location
			new_locations = [cols_dict[i] for i in curr_cols]

			# return new kmer_matrix with cols in the correct spot
			curr_matrix = curr_matrix[:,np.argsort(new_locations)]
			curr_cols = curr_cols[np.argsort(new_locations)]

			# load booster
			bst = joblib.load("predict/models/xgb_public_{}feats_{}model.bst".format(str(features),drug))

			# we can pull the new features list from the xgboost booster to ensure the features are in the same order
			booster_names = bst.feature_names

			# create a dmatrix for testing
			dtest = xgb.DMatrix(curr_matrix,feature_names=curr_cols)

			# verifying that the columns we are loading in are in the same order as when the model was trained
			for train_kmer, test_kmer in zip(drug_feats,curr_cols):
				try:
					assert(train_kmer==test_kmer)
				except:
					print("Expected kmer {} but got {}".format(train_kmer,test_kmer))
					raise

			# if your GPU has a compute capability below 3.5 this next line will fail
			import time
			start = time.time()
			predictions = [int(round(i)) for i in bst.predict(dtest, validate_features = True)]
			end = time.time()
			#print("Prediction on {} took {}s".format(drug, end-start))

			# predictions will be encoded into 0,1,2,3... so we need to bring them back to MIC values
			encoder = { i :  mic_class_dict[drug][i] for i in range(0, len(mic_class_dict[drug]))}
			predictions = np.array([encoder[i] for i in predictions])

			#le = preprocessing.LabelEncoder()
			#le.classes_ = np.asarray(mic_class_dict[drug])
			#predictions = le.inverse_transform(predictions)

			# if there is a column for this drug, we need to make predictions for it
			has_mic_labels = False
			if(evaluate):
				try:
					drug_col = mic_df[drug]
					has_mic_labels = True
				except:
					print("No valid column heading for {}, skipping evaluation.")


			# now we put them into the dataframe to be saved
			assert(len(predictions)==len(kmer_rows))

			# for every genome we predicted on, we need to add the results to our DataFrame
			# and then we need to evaluate the prediction for errors
			missing_counter = 0
			for prediction, run_id in zip(predictions, kmer_rows):
				predict_df.at[run_id,drug] = prediction

				try:
					# check to see if this genome has mic values for this drug
					genome_index = list(mic_df.index).index(run_id)

					# find what the correct prediction would be
					actual = drug_col[genome_index]
					assert(actual in mic_class_dict[drug])

					off_by_one = False

					# find indexing so we can compare without it being a log increase
					act_loc = mic_class_dict[drug].index(actual)
					pred_loc = mic_class_dict[drug].index(prediction)

					if(pred_loc == act_loc):
						# if we are in here, there was no error so we can just say that so we know how many predictions were evaluated
						with open('predict/prediction_errors.txt', 'a') as myfile:
							myfile.write("\nCorrect:"+drug)
						continue

					if(pred_loc==act_loc+1 or pred_loc==act_loc-1):
						#checking if the prediction was within 1 dilution of the actual value
						off_by_one = True

					# now we write all of the error information to file
					with open('predict/prediction_errors.txt', 'a') as myfile:
						myfile.write("\nDrug:{} Genome:{} Predicted:{} Actual:{} OBO:{} Major?:{}".format(drug, run_id, prediction, actual, off_by_one,find_major(pred_loc,act_loc,drug,mic_class_dict)))
				except:
					#print("Could not find a valid {} MIC for {}".format(drug, run_id))
					missing_counter +=1
					continue
			print("{}/{} {} samples did not have valid labels, see predict/logs/mic_public.log for more info".format(missing_counter,len(predictions),drug))



		# save the dataframe of predictions as a .csv
		predict_df.to_csv("predict/predictions.csv")

		if(evaluate):
			shell("python src/genome_error_table_converter.py predict/prediction_errors.txt")

			# create a dataframe for the results to be stored in as we read them
			results_df = pd.DataFrame(data = np.zeros((len(drugs),9)),index = drugs,columns=[
			'Accuracy (1D)','Accuracy (Direct)','Total Predictions','Non-Major Error Rate',
			'Major Error Rate','Very Major Error Rate','Non-Major Error Rate (1D)',
			'Major Error Rate (1D)','Very Major Error Rate (1D)'])

			# we are going to walk line by line classifying results
			with open("predict/prediction_errors.txt") as file:
				for line_num, line in enumerate(file):

					# looking at the first 5 characters we can see if the prediction was correct or if it was an error
					first_word = line[:5]

					# if it was an error we need to collect what type it was
					if(first_word=='Drug:'):

						# this is the ordering of how the errors are written to the file
						drug, genome, pred, act, off_by_one, major = line.split(' ')

						# we need to split off all the english to just get the keyvalue
						drug = drug.split(':')[1]
						off_by_one = off_by_one.split(':')[1]
						major = major.split(':')[1]

						# lines end with a \n that we need to remove
						major = major.rstrip()

						# based on what we find we need to increment the correct value in the dataframe, we will convert to % later
						results_df.at[drug,'Total Predictions']+=1
						if(off_by_one=='True'):
							results_df.at[drug,'Accuracy (1D)']+=1
							if(major=='NonMajor'):
								results_df.at[drug,'Non-Major Error Rate']+=1
							elif(major=='MajorError'):
								results_df.at[drug,'Major Error Rate']+=1
							elif(major=='VeryMajorError'):
								results_df.at[drug,'Very Major Error Rate']+=1
							else:
								raise Exception("Major rate: {} not able to be properly classified".format(major))
						elif(major=='NonMajor'):
							results_df.at[drug,'Non-Major Error Rate (1D)']+=1
						elif(major=='MajorError'):
							results_df.at[drug,'Major Error Rate (1D)']+=1
						elif(major=='VeryMajorError'):
							results_df.at[drug,'Very Major Error Rate (1D)']+=1
						else:
							raise Exception("Major rate: {} not able to be properly classified".format(major))

					# if the prediction was correct we simply increment both accuracies
					elif(first_word=='Corre'):
						drug = line.split(':')[1]
						drug = drug.rstrip()
						results_df.at[drug,'Total Predictions']+=1
						results_df.at[drug,'Accuracy (1D)']+=1
						results_df.at[drug,'Accuracy (Direct)']+=1

					elif(first_word=='\n'):
						# this is intentionally left empty to skip newline chars
						continue

					else:
						raise Exception("An unexpected {} was found in prediction_errors on line {}".format(first_word,line_num+1))

			# now we want to generate rates from raw numbers
			for drug in drugs:
				# for accuracies we divide raw counts by total predictions
				total = results_df['Total Predictions'][drug]
				results_df.at[drug,'Accuracy (1D)'] = results_df['Accuracy (1D)'][drug] / total
				results_df.at[drug,'Accuracy (Direct)'] = results_df['Accuracy (Direct)'][drug] / total

				for error_type in ['Non-Major Error Rate','Major Error Rate','Very Major Error Rate']:
					results_df.at[drug, error_type] += results_df.at[drug, error_type+' (1D)']

				# for errors we divide raw counts by total predictions
				num_errors = 0
				for error_type in ['Non-Major Error Rate','Major Error Rate','Very Major Error Rate',
				'Non-Major Error Rate (1D)','Major Error Rate (1D)','Very Major Error Rate (1D)']:
					num_errors+= results_df[error_type][drug]
					results_df.at[drug, error_type] = results_df[error_type][drug] / total


			results_df.to_csv("predict/results.csv")
