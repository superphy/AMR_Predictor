"""
Takes in Precision recall table and returns occurrences
of each top gene in each class
"""
import numpy as np
import pandas as pd
import os, sys
import glob
from collections import Counter
from sklearn.externals import joblib

drugs=["AMP","AMC","AZM","CHL","CIP","CRO","FIS","FOX","GEN","NAL","SXT","TET","TIO"]
datasets=["public","grdi"]
kmer_length = '31'

def load_top_kmers(drug, dataset):
    if(kmer_length == '11'):
        top = np.load("annotation/search/features_scores/{}_{}.npy".format(drug, dataset))
        return ["{} ({:.2%})".format(i[0],float(i[1])) for i in top]
    else:
        top = np.load("data/multi-mer/feat_ranks/{}_1000000_{}_{}mer_top5_feats.npy".format(
        dataset, drug, kmer_length), allow_pickle = True)[:,0]

        ranks = np.load("data/multi-mer/feat_ranks/{}_1000000_{}_{}mer_feature_ranks.npy".format(
        dataset, drug, kmer_length), allow_pickle = True)

        scores = []

        for kmer in top:
            score = ranks[1][np.where(ranks[0]==kmer)]
            scores.append("{} ({:.2%})".format(kmer,float(score)))

    return scores

def get_counts(kmers, drug, dataset, classes):
    if dataset == 'grdi':
        path = 'grdi_'
    else:
        path = ''
    if kmer_length == '11':
        matrix = np.load("data/filtered/{}{}/kmer_matrix.npy".format(path, drug))
        rows_mic = np.load("data/filtered/{}{}/kmer_rows_mic.npy".format(path, drug))
        cols = np.load("data/unfiltered/kmer_cols.npy")
    else:
        # already filtered to the top million
        matrix = np.load("data/multi-mer/kbest/{}/1000000_{}_{}mer_matrix.npy".format(dataset, drug, kmer_length))
        cols = np.load("data/multi-mer/kbest/{}/1000000_{}_{}mer_cols.npy".format(dataset, drug, kmer_length))
        Z = np.load("data/multi-mer/{}{}mer_rows.npy".format(path, kmer_length))

        mic_df = joblib.load("data/{}_mic_class_dataframe.pkl".format(dataset))
        mic_class_dict = joblib.load("data/{}_mic_class_order_dict.pkl".format(dataset))

        Y = [mic_df[drug][i] for i in Z]
        row_mask = [i in mic_class_dict[drug] for i in Y]

        rows_mic = np.asarray(Y)[row_mask]

    if isinstance(cols[0],bytes):
        cols = [i.decode('utf-8') for i in cols]

    if isinstance(classes[0],bytes):
        classes = [i.decode('utf-8') for i in classes]

    if isinstance(rows_mic[0],bytes):
        rows_mic = [i.decode('utf-8') for i in rows_mic]

    rows_dict = { rows_mic[i] : i for i in range(0, len(rows_mic))}
    cols_dict = { cols[i] : i for i in range(0, len(cols))}

    counts_dict = {}

    # currently counts frequency of genomes with >0 hits, not number of hits
    for kmer in kmers:
        counts = pd.Series(data = np.zeros(len(classes)), index = classes, dtype = 'object')
        seen = []
        for i, count in enumerate(matrix[:,cols_dict[kmer]]):
            if count > 0:
                seen.append(rows_mic[i])
        kmer_freq = Counter(seen)

        for class_id in classes:
            counts[class_id] = kmer_freq[class_id]
        counts_dict[kmer] = counts

    return counts_dict

def add_percent(supports, kmer_counts, classes):
    data = ["{} ({:.2%})".format(kmer_counts[i],kmer_counts[i]/supports[i]) for i in classes]
    return pd.Series(data = data, index = classes)

if __name__ == "__main__":

    writer = pd.ExcelWriter('results/kmer_per_class/{}mer_frequency_per_class.xlsx'.format(kmer_length))
    for drug in drugs:
        for dataset in datasets:
            if dataset == 'grdi' and drug in ['AZM','FIS']:
                continue
            if(kmer_length == '11'):
                input = glob.glob('results/*/{0}_1000feats_XGBtrainedOn{1}_testedOnaCrossValidation.pkl'.format(drug, dataset))[0]
            else:
                input = "results/multi-mer/{}mer/kbest/{}_0feats_XGBtrainedOn{}_testedOnaCrossValidation.pkl".format(kmer_length, drug, dataset)
            result_df = pd.read_pickle(input)
            top_kmers = load_top_kmers(drug,dataset)

            names = [i.split()[0] for i in top_kmers]
            classes = result_df.index.values

            count_dict =  get_counts(names, drug, dataset, classes)

            for kmer in names:
                result_df[kmer] = add_percent(result_df['Supports'],count_dict[kmer], classes)

            result_df = result_df.drop(columns = ['1D Acc'])

            if dataset == 'grdi':
                dname = 'GRDI'
            else:
                dname = 'NCBI'
            result_df.to_excel(writer, "{} {}".format(dname,drug))
    writer.save()
