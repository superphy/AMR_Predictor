

""" PIPELINE
clean
count - Jellyfish
dump - jellyfish
predict - load into matrix, spit out csv(from pandas)
evaluate - if they have MIC's listed, evaluate the model
"""
ids, = glob_wildcards("genomes/raw/{id}.fasta")
rule all:

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
        "jellyfish dump {input} > {output}"

rule matrix:
    # This rule is preprocessing the data to be in the correct format to be fed
    # into the machine learning models
    input:
        expand("predict/genomes/jellyfish_results/{id}.fa", id=ids)
    output:
        "predict/genomes/unfiltered/kmer_matrix.npy",
        "predict/genomes/unfiltered/kmer_rows.npy",
        "predict/genomes/unfiltered/kmer_cols.npy"
    run:
        import os, sys
        import numpy
        from Bio import Seq, SeqIO

        # find all the possible features that are going to be used to make the prediction
        relevant_feats = []
        for feat_array in ([files for r,d,files in os.walk("predict/features/")][0]):
            relevant_feats = np.concatenate((relevant_feats, np.load("predict/features/"+feat_array)))

        # remove any duplicates
        relevant_feats = [i for i in set(relevant_feats)]

        # find all the genomes we were given, genomes are filenames and runs are sample names
        genomes = ([files for r,d,files in os.walk("predict/genomes/jellyfish_results/")][0])
        runs = [i.split('.')[0] for i in genomes]

        # declaring empty kmer matrix to fill
        kmer_matrix = np.zeros((len(genomes),len(relevant_feats)),dtype = 'uint8')

        # making dicts for faster indexing
        rows_dict = { genomes[i] : i for i in range(0, len(genomes))}
        cols_dict = { relevant_feats[i] : i for i in range(0, len(relevant_feats))}

        # go through each genome
        for genome in genomes:
            for record in SeqIO.parse(genome, "fasta"):
                kmer_seq = kmerseq._get_seq_str_and_check_alphabet(record.seq)
                kmer_count = int(record.id)

                # load kmer matrix at the correct cell if its a relevant kmer
                if(kmer_seq in relevant_feats):
                    kmer_matrix[rows_dict[genome]][cols_dict[kmer_seq]]=kmer_count

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
        from data_transformers import remove_symbols

        drugs = ['AMP','AMC','AZM','CHL','CIP','CRO','FIS','FOX','GEN','NAL','SXT','TET','TIO']

        # predictions will be encoded into 0,1,2,3... so we need to bring them back to MIC values
        le = preprocessing.LabelEncoder()
        mic_class_dict = joblib.load("data/public_mic_class_order_dict.pkl")

        # load in 2D matrix of kmer counts, columns are kmers and rows are genomes so
        # kmer_matrix[genome][kmer] returns how many times that kmer was seen in that genome
        kmer_matrix = np.load("predict/genomes/unfiltered/kmer_matrix.npy")

        # load labels from last ruls
        kmer_rows = np.load("predict/genomes/unfiltered/kmer_rows.npy")
        kmer_cols = np.load("predict/genomes/unfiltered/kmer_cols.npy")

        # make pandas to store predictions
        predicts = np.zeros((len(kmer_rows), 13),dtype= 'object')
        predict_df = pd.DataFrame(data = predicts, index = kmer_rows, columns = drugs)

        # go through each drug and make prediction for just that drug
        for drug in drugs:
            # just use the data relevant to this drug
            drug_feats = np.load("predict/features/1000feats_"+drug+".npy")
            cols_dict = { kmer_cols[i] : i for i in range(0, len(kmer_cols))}
            feat_mask = [i in drug_feats for i in kmer_cols]

            # transpose, apply mask, transpose again
            curr_matrix = np.transpose(np.transpose(kmer_matrix)[feat_mask])
            curr_cols = kmer_cols[feat_mask]

            # now we have the correct columns, but they are most likely in the wrong order
            # for every kmer in current order, find new location
            new_locations = [cols_dict[i] for i in curr_cols]

            # return new kmer_matrix with cols in the correct spot
            curr_matrix = curr_matrix[:,np.argsort(new_locations)]

            # load model
            model = pickle.load(open("xgb_public_1000feats_{}model.dat".format(drug)."wb"))
            predictions = [round(i) for i in model.predict(curr_matrix)]

            # predictions are currently encoded in 0,1,2,3,4 and we need MIC values'
            mic_dict = [remove_symbols(i) for i in mic_class_dict[drug]]
            le.fit(mic_dict)
            predictions = le.inverse_transform(predictions)

            # now we put them into the dataframe to be saved
            assert(len(prediction)==len(kmer_rows))
            for prediction, run_id in zip(predictions, kmer_rows):
                predict_df.at[run_id,drug] = prediction

        # save the dataframe of predictions as a .csv

        predict_df.to_csv("predict/predictions.csv")
