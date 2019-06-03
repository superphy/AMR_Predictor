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
        expand("annotation/{drug}_"+dataset+"_feature_ranks.npy", drug = drugs)

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
        "python annotation/15mer_matrix.py {input.a} {params.drug}"

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
        "python annotation/15mer_model.py {params.drug} {threads} "+dataset
