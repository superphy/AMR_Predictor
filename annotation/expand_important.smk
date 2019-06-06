"""
This snakemake is to expand the top X 11-mers into all possible 15-mers to train a model.
Top features need to be saved for each drug
"""
drugs=["AMP","AMC","AZM","CHL","CIP","CRO","FIS","FOX","GEN","NAL","SXT","TET","TIO"]
dataset=["public","grdi"][0]

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

ids, = glob_wildcards("data/genomes/clean/{id}.fasta")

rule all:
    input:
        "annotation/15mer_annotation_summary.csv"

rule count:
    input:
        "data/genomes/clean/{id}.fasta"
    output:
        temp("data/jellyfish_results15/{id}.jf")
    threads:
        2
    shell:
        "jellyfish count -C -m 15 -s 100M -t {threads} {input} -o {output}"

rule dump:
    input:
        "data/jellyfish_results15/{id}.jf"
    output:
        "data/jellyfish_results15/{id}.fa"
    shell:
        "jellyfish dump {input} > {output}"

rule expand:
    output:
        "annotation/{drug}_1000feats_"+dataset+"_15mers.npy"
    params:
        drug = "{drug}"
    threads:
        5
    shell:
        "python annotation/expand_perms.py {params.drug} "+dataset

rule matrix:
    input:
        a = expand("annotation/{drug}_1000feats_"+dataset+"_15mers.npy", drug = drugs),
        b = expand("data/jellyfish_results15/{id}.fa", id=ids)
    output:
        "annotation/15mer_data/{drug}_kmer_matrix.npy",
        "annotation/15mer_data/{drug}_kmer_rows.npy",
        "annotation/15mer_data/{drug}_kmer_cols.npy"
    params:
        drug = "{drug}"
    shell:
        "sbatch -c 48 --mem 32G --partition NMLResearch --wrap='python annotation/15mer_matrix.py {input.a} {params.drug}'"

rule model:
    input:
        expand("annotation/15mer_data/{drug}_kmer_matrix.npy", drug = drugs)
    output:
        "annotation/{drug}_"+dataset+"_feature_ranks.npy"
    params:
        drug = "{drug}"
    threads:
        16
    shell:
        "sbatch -c 16 --mem 125G --partition NMLResearch --wrap='python annotation/15mer_model.py {params.drug} {threads} "+dataset+"'"

rule prokka:
    input:
        GENOMES_PATH+"{id}.fasta"
    output:
        "annotation/annotated_genomes/{id}/{id}.ffn",
    threads:
        1
    shell:
        "prokka {input} --outdir annotation/annotated_genomes/{wildcards.id} --prefix {wildcards.id} --cpus {threads} --force --compliant"

rule single_line:
    input:
        "annotation/annotated_genomes/{id}/{id}.ffn"
    output:
        "annotation/annotated_genomes/{id}/{id}.ffns"
    shell:
        "awk \'!/^>/ {{ printf \"%s\", $0; n = \"\\n\" }} ; /^>/ {{ print n $0; n = \"\" }} ; END {{ printf \"%s\", n }}\' < {input}  > {output}"

rule grep_feats:
    input:
        a = expand("annotation/annotated_genomes/{id}/{id}.ffns", id = ids),
        b = expand("annotation/{drug}_"+dataset+"_feature_ranks.npy", drug = drugs)
    output:
        'annotation/gene_hits_15mer_for_{drug}_1000feats.out'
    params:
        drug = "{drug}"
    run:
        import numpy as np
        top_model_feats = []
        top_stats_feats = []

        # all_feats is a 2D array with 3 rows: 15mer, model_importance,chi2 importance
        all_feats = np.load(input.a)

        # pull top 5 feats and search through genes for hits
        for imp_measure in [1,2]:

            # we want to pull at most 5 features
            top_x_feats = 5
            while(top_x_feats>0):
                # find the highest scoring value
                m = max(all_feats[imp_measure])

                # pull the index of all kmers tieing the top score
                top_indeces = [i for i, j in enumerate(all_feats[imp_measure]) if j == m]

                # make sure we dont take more than top_x total, this can be removed
                # to keep all tying features in the pipeline but beware, if all are zero it will search through a thousand kmers
                if(len(top_indeces) > top_x_feats):
                    top_indeces[:top_x_feats]

                top_x_feats -= len(top_indeces)
                for i in top_indeces:
                    # add the kmer to correct list, and then set its value to zero for the next round
                    if(imp_measure == 1):
                        top_model_feats.append((all_feats[0][i]))
                    else:
                        top_stats_feats.append((all_feats[0][i]))
                    all_feats[imp_measure][i] = 0

            # double check that we have the correct number of features
            assert(len(top_model_feats)<=5 and len(top_stats_feats)<=5)

            for seq in top_feats:
                for root, dirs, files in os.walk("annotation/annotated_genomes/"):
                    for genome in dirs:
                        shell("echo '\n\nSearching_for_{seq}_in_{genome}_{params.drug}{imp_measure}' >> {output} && grep -B 1 {seq} annotation/annotated_genomes/{genome}/{genome}.ffns >> {output} || echo 'not found' >> {output}")
rule summary:
    input:
        expand('annotation/gene_hits_15mer_for_{drug}_1000feats.out', drug = drugs)
    output:
        "annotation/15mer_annotation_summary.csv"
    params:
        drug = "{drug}"
    run:
        # columns for pandas dataframe, this will be faster than appending to the pandas
        df_drug = []
        df_OxF_mer = []
        df_OxB_mer = []
        df_hit = []
        df_imp = []

        for drug in drugs:
            # feature importances
            all_feats = np.load("annotation/{params.drug}_"+dataset+"_feature_ranks.npy")

            # array of 15mers, grouped by parent
            OxF_mers = np.load("annotation/{}_1000feats_public_15mers.npy".format(params.drug))

            # array of 11mer parents
            OxB_mers = np.load("annotation/{}_1000feats_public_15mers_parent.npy".format(params.drug))

            # load gene hits to prep for creation of pandas dataframe
            with open("annotation/gene_hits_15mer_for_{}_1000feats.out".format(params.drug)) as file:

                # primers, so when we get a hit we know what the informatin was about it
                OxF_mer = ''
                genome = ''
                drug = ''
                importance_measure = ''

                # go through each gene search
                for line in file:

                    # prime the next loading sequence
                    if(line[:6]=='Search'):
                        intro,f,OxF_mer,n,genome,drug_imp = line.split('_')
                        drug_id = drug_imp[:3]
                        importance_measure = drug_imp[-1]

                    # if we found a gene, load the primers into the pre-pandas lists at the top of this snakemake run block
                    if(line[0] == '>'):
                        # pull the name of the gene that was hit
                        tag = line.split(' ',1)[1]

                        # append each of the 5 things requires to build the matrix
                        df_drug.append(drug_id)
                        df_OxF_mer.append(OxF_mer)

                        # find the index of the parent 11mer and append it
                        for parent_num, OxF_mer_expansion in enumerate(OxF_mers):
                            if OxF_mer in OxF_mer_expansion:
                                df_OxB_mer.append(OxB_mers[parent_num])
                                break
                            if(parent_num == 4):
                                raise Exception("Could not find parent 11mer for the 15mer", OxF_mer)
                        df_hit.append(tag)
                        df_imp.append(importance_measure)

        current_drug = ''
        curr_start_index = 0
        curr_stop_index = 0
        for i, pre_pandas_drug in enumerate(df_drug):

            # if we are starting a new drug
            if pre_pandas_drug != current_drug:
                curr_stop_index = i-1

                # right now we know that current_drug runs from index
                # curr_start to index curr_stop

                # making sure we arent at the start of the loop
                if(i!=0):
                    model_15mers = []
                    stat_15mers = []


                    #TODO:
                    # go through the start and stop index and then build a dictionary for kmers/genes and how many hits they get
                    # then generate a row for each so that we have Drug - 15mer - 11mer - geneFound - numHitsForThatGene - sourceOfImportance (chi2vsXGBoost)
                    # maybe not in that order though
                    # append it to some nested master list so we can make a pandas dataframe to save as a csv

                curr_start_index = i
                current_drug = pre_pandas_drug
