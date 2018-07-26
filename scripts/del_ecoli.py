import pandas as pd
import os

if __name__ == '__main__':
	"""
	Deletes E. coli genome fasta files from the genomes/raw folder.
	"""
	genome_list = []
	mic_df = pd.ExcelFile('../amr_data/Updated_GenotypicAMR_Master.xlsx').parse('GenotypicAMR_Master')
	df_rows = mic_df.index.values
	num_rows = len(df_rows)
	for index, row in mic_df.iterrows():
		if row["genus"] == "Escherichia":
			genome_list.append(row["run"])
	#print(genome_list)
	#print(len(genome_list))
	for genome in genome_list:
		#os.remove('/home/CSCScience.ca/jmoat/genomes/raw/'+genome+'.fasta')
		os.remove('../genomes/raw/'+genome+'.fasta')