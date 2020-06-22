"""
Takes in all the gene annotation information and chooses the best gene hits per kmer
"""

import pandas as pd
import numpy as np
import os, sys

drugs=["AMP","AMC","AZM","CHL","CIP","CRO","FIS","FOX","GEN","NAL","SXT","TET","TIO"]
datasets=["public","grdi"]

# list of top 5 kmers and their scores for every antimicrobial (contains grdi and ncbi

def find_top_genes(kmer_length, search_num):
    if(kmer_length =='11'):
        kmer_ranks = pd.read_csv("annotation/search/features_scores/score_summary.csv")
    else:
        kmer_ranks = pd.read_csv("data/multi-mer/feat_ranks/{}mer_score_summary.csv".format(kmer_length))

    #top_genes = pd.DataFrame(data=np.zeros((1,10), dtype='object'), columns=['dataset','drug','kmer','gene','name','count','average_dist', 'std_dev' ,"AMG", "AMG's Only"])
    top_genes = pd.DataFrame(data=np.zeros((1,9), dtype='object'), columns=['dataset','drug','kmer','gene','name','count','average_dist', 'std_dev' ,"AMG"])

    for dataset in datasets:
        if(kmer_length == '11'):
            gene_hits = pd.read_csv("annotation/search/11mer_data/{}_11mer_summary.csv".format(dataset))
        else:
            gene_hits = pd.read_csv("results/multi-mer/{0}mer/{1}_{0}mer_summary.csv".format(kmer_length,dataset))


        dataset_ranks = kmer_ranks[kmer_ranks['Dataset']==dataset]

        for drug in drugs:
            if(dataset == 'grdi' and drug in ['FIS']):
                continue

            """
            # find top kmer
            drug_kmers = dataset_ranks[dataset_ranks['Antimicrobial']==drug]

            drug_kmers = drug_kmers.reset_index(drop=True)

            # returns the top kmer according to the score summary
            top_kmer = drug_kmers["{}mer".format(kmer_length)][0]

            # find AMG with the most hits
            all_hits = gene_hits[(gene_hits['drug']==drug) & (gene_hits['kmer']==top_kmer)]

            # sort for only genes of known antimicrobial resistance
            amg_hits = all_hits[(all_hits['AMG']!='0')&(all_hits['AMG']!=0)]
            amg_hits = amg_hits.reset_index(drop=True)

            # merge rows with same gene name after removing '_'

            #for search_num, search_space in enumerate([amg_hits,all_hits]):
            for search_num, search_space in enumerate([all_hits]):
                genes = list(search_space['gene'])
                counts = list(search_space['count'])

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

                for i in range(search_space.shape[0]):
                    search_space.at[i,'count']=counts[i]

                search_space = search_space[search_space['count']!= 0]
                search_space = search_space.drop(columns = 'Unnamed: 0')

                # check which has the highest number of hits, report that.
                top_hit = search_space[search_space['count']==search_space['count'].max()]
                #top_hit["AMG's Only"]=pd.Series(search_num==0)

                top_genes = pd.concat([top_genes,top_hit])
            """
            # find top kmer
            drug_kmers = dataset_ranks[dataset_ranks['Antimicrobial']==drug]

            drug_kmers = drug_kmers.reset_index(drop=True)

            # returns the top kmer according to the score summary
            top_kmer = drug_kmers["{}mer".format(kmer_length)][0]

            # find AMG with the most hits
            all_hits = gene_hits[(gene_hits['drug']==drug) & (gene_hits['kmer']==top_kmer)]

            # Perhaps if we want to run for more than the top kmer, we make everything below
            # this a function and then call it 5 times, returning the highest.

            # sort for only genes of known antimicrobial resistance
            amg_hits = all_hits[(all_hits['AMG']!='0')&(all_hits['AMG']!=0)]

            # merge rows with same gene name after removing '_'

            #for search_num, search_space in enumerate([amg_hits,all_hits]):
            if search_num == 0:
                search_space = amg_hits
            else:
                search_space = all_hits
            search_space = search_space.reset_index(drop=True)
            genes = list(search_space['gene'])
            counts = list(search_space['count'])

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

            for i in range(search_space.shape[0]):
                search_space.at[i,'count']=counts[i]

            search_space = search_space[search_space['count']!= 0]
            search_space = search_space.drop(columns = 'Unnamed: 0')

            # check which has the highest number of hits, report that.
            top_hit = search_space[search_space['count']==search_space['count'].max()]
            #top_hit["AMG's Only"]=pd.Series(search_num==0)

            top_genes = pd.concat([top_genes,top_hit])


    return top_genes

if __name__ == "__main__":
    kmer_length = sys.argv[1]
    if(kmer_length == '11'):
        save_path = "annotation/search/11mer_data/top_hits_per_drug.csv"
    else:
        save_path = "results/multi-mer/{}mer/best_hits.csv".format(kmer_length)
    amg_genes = find_top_genes(kmer_length, 0)
    print(amg_genes)
    all_genes = find_top_genes(kmer_length, 1)
    print(all_genes)

    pd.concat([amg_genes,all_genes]).to_csv(save_path)
