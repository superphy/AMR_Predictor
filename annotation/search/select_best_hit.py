"""
Takes in all the gene annotation information and chooses the best gene hits per kmer
"""

import pandas as pd
import numpy as np
import os, sys

drugs=["AMP","AMC","AZM","CHL","CIP","CRO","FIS","FOX","GEN","NAL","SXT","TET","TIO"]
datasets=["public","grdi"]

# list of top 5 kmers and their scores for every antimicrobial (contains grdi and ncbi)
kmer_ranks = pd.read_csv("annotation/search/features_scores/score_summary.csv")

top_genes = pd.DataFrame(data=np.zeros((1,9), dtype='object'), columns=['dataset','drug','kmer','gene','name','count','average_dist', 'std_dev' ,"AMG"])

for dataset in datasets:
    gene_hits = pd.read_csv("annotation/search/11mer_data/{}_11mer_summary.csv".format(dataset))
    dataset_ranks = kmer_ranks[kmer_ranks['Dataset']==dataset]

    for drug in drugs:
        if(dataset == 'grdi' and drug in ['AZM', 'FIS']):
            continue

        # find top kmer
        drug_11mers = dataset_ranks[dataset_ranks['Antimicrobial']==drug]

        drug_11mers = drug_11mers.reset_index(drop=True)

        top_kmer = drug_11mers['11mer'][0]

        # find AMG with the most hits
        amg_hits = gene_hits[(gene_hits['drug']==drug) & (gene_hits['kmer']==top_kmer)]

        # sort for only genes of known antimicrobial resistance
        amg_hits = amg_hits[(amg_hits['AMG']!='0')&(amg_hits['AMG']!=0)]

        amg_hits = amg_hits.reset_index(drop=True)

        # merge rows with same gene name after removing '_'

        genes = list(amg_hits['gene'])
        counts = list(amg_hits['count'])

        # first time gene seen, 2nd val is count, 2nd+ time seen, count is 0
        for i in range(len(genes)):
            gene = genes[i]
            if(gene == 'del' or i==0):
                continue

            # if the gene was seen above
            if(gene.split('_')[0] in [i.split('_')[0] for i in genes[:i]]):
                genes[i] = 'del'
                counts[[i.split('_')[0] for i in genes[:i]].index(gene.split('_')[0])]+=counts[i]
                counts[i] = 0

        for i in range(amg_hits.shape[0]):
            amg_hits.at[i,'count']=counts[i]

        amg_hits = amg_hits[amg_hits['count']!= 0]
        amg_hits = amg_hits.drop(columns = 'Unnamed: 0')

        # check which has the highest number of hits, report that.
        # top_genes.append(amg_hits[amg_hits['count']==amg_hits['count'].max()])
        top_genes = pd.concat([top_genes,amg_hits[amg_hits['count']==amg_hits['count'].max()]])

print(top_genes)

top_genes.to_csv('annotation/search/11mer_data/top_hits_per_drug.csv')
