"""
For each antimicrobial, this program takes the top 5 11mers
and a list of possible 15mer expansions, and finally returns
the most important of each of the 15mers as according to a chi2.

For each 5 top-11mers, we search through the genome using the top 15mer

This script requires the outputs from expand_important.smk
"""

import numpy as np
import pandas as pd
import os, sys

drugs=["AMP","AMC","AZM","CHL","CIP","CRO","FIS","FOX","GEN","NAL","SXT","TET","TIO"]

def nan_to_zero(ele):
    if(ele=='nan'):
        return 0
    return ele

rule all:
    input:
        "annotation/top15mer_per_11mer_summary.csv"

rule grep_feats:
    output:
        "annotation/top_11mer_gene_search_{drug}.out"
    params:
        drug = "{drug}"
    run:
        drug = params.drug

        # 2D array, each row consists of 1280 15mer expansions
        OxF_mers = np.load("annotation/{}_1000feats_public_15mers.npy".format(drug))

        # 1D array, 11mer parent labels for each row of 15mers
        # OxF_mer_parents[0] is the 11mer parent of all 15mers in OxF_mers[0]
        OxF_mer_parents = np.load("annotation/{}_1000feats_public_15mers_parent.npy".format(drug))

        # 2D array of shape (3,6400)
        # row[0] are 15mer, row[1] is the score by xgboost, row[2] is the score by chi2
        feature_ranks = np.load("annotation/{}_public_feature_ranks.npy".format(drug))

        # set 'nan' strings to zero for max functions
        feature_ranks[2] = [nan_to_zero(i) for i in feature_ranks[2]]

        # loop throug each 11mer parent, find the highest performing 15mer
        for par_indx, parent in enumerate(OxF_mer_parents):

            # only look through the 1280 15mers relating to that parent 11mer
            curr_feat_ranks = feature_ranks[:,(1280*par_indx):(1280*(par_indx+1))]

            # double check we did everything correct
            assert(len(curr_feat_ranks[0])==1280)
            for i, mer in enumerate(OxF_mers[par_indx]):
                assert(mer==curr_feat_ranks[0][i])

            # find the top 15mer by the chi2
            m = max(curr_feat_ranks[2])

            # pull the index of all kmers tieing the top score
            top_indeces = [i for i, j in enumerate(curr_feat_ranks[2]) if j == m]

            # make sure we dont take more than 1 total, takes the first in the event of a tie
            if(len(top_indeces) > 0):
                top_indeces = top_indeces[0]
            top_15mer = curr_feat_ranks[0][top_indeces]
            top_15mer_score = curr_feat_ranks[2][top_indeces]


            for root, dirs, files in os.walk("annotation/annotated_genomes/"):
                for genome in dirs:
                    shell("echo '\n\nSearching_for_{top_15mer}_in_{genome}_{drug}' >> {output} && grep -B 1 {top_15mer} annotation/annotated_genomes/{genome}/{genome}.ffns >> {output} || echo 'not found' >> {output}")

rule summary:
    input:
        expand("annotation/top_11mer_gene_search_{drug}.out", drug = drugs)
    output:
        "annotation/top15mer_per_11mer_summary.csv"
    run:
        # columns for pandas df (faster than appending to pd.df)
        df_drug = []
        df_OxF_mer = []
        df_OxB_mer = []
        df_OxF_score = []
        df_gene = []

        dataset = 'public'

        for drug in drugs:
            # feature importances
            all_feats = np.load("annotation/{}_".format(drug)+dataset+"_feature_ranks.npy")

            # array of 15mers, grouped by parent
            OxF_mers = np.load("annotation/{}_1000feats_public_15mers.npy".format(drug))

            # array of 11mer parents
            OxB_mers = np.load("annotation/{}_1000feats_public_15mers_parent.npy".format(drug))

            # load gene hits to prep for creation of pandas dataframe
            with open("annotation/top_11mer_gene_search_{}.out".format(drug)) as file:

                # primers, so when we get a hit we know what the informatin was about it
                OxF_mer = ''
                genome = ''
                drug = ''
                importance_measure = ''

                # go through each gene search
                for line in file:

                    # prime the next loading sequence
                    if(line[:6]=='Search'):
                        intro,f,OxF_mer,n,genome,drug = line.split('_')
                        drug = drug.rstrip()

                    # if we found a gene, load the primers into the pre-pandas lists at the top of this snakemake run block
                    if(line[0] == '>'):
                        # pull the name of the gene that was hit
                        tag = line.split(' ',1)[1]

                        # append each of the 5 things requires to build the matrix
                        df_drug.append(drug)
                        df_OxF_mer.append(OxF_mer)

                        # find the index of the parent 11mer and append it
                        for parent_num, OxF_mer_expansion in enumerate(OxF_mers):
                            if OxF_mer in OxF_mer_expansion:
                                df_OxB_mer.append(OxB_mers[parent_num])
                                break
                            if(parent_num == 4):
                                raise Exception("Could not find parent 11mer for the 15mer", OxF_mer)
                        df_gene.append(tag)
                        df_OxF_score.append(all_feats[2][list(all_feats[0]).index(OxF_mer)])


        current_drug = ''
        curr_start_index = 0
        curr_stop_index = 0
        all_hits = []

        for i, pre_pandas_drug in enumerate(df_drug):
            all_hits.append("{}_{}_{}_{}_{}".format(df_drug[i],df_OxF_mer[i],df_OxB_mer[i],df_gene[i],df_OxF_score[i]))
        from collections import Counter
        hit_counts = dict(Counter(all_hits))

        pre_pandas = np.zeros((len(hit_counts),6),dtype='object')

        for i, (key, value) in enumerate(hit_counts.items()):
            #count_drug, count_15mer, count_11mer, count_hits, count_imp = key.split('_')
            splits = key.split('_')
            pre_pandas[i][5] = value
            for j, split in enumerate(splits):
                pre_pandas[i][j] = split

        import pandas as pd

        final_df = pd.DataFrame(data=pre_pandas,columns=["Drug","15mer","Parent 11mer","Gene","Chi2 Score of 15mer","Number of Hits",])

        final_df.to_csv("annotation/top15mer_per_11mer_summary.csv")
