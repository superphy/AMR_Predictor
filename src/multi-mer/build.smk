ids, = glob_wildcards("data/genomes/raw/{id}.fasta")
gids, = glob_wildcards("data/grdi_genomes/raw/{gid}.fasta")

kmer_length = '31'
set_nums = [str(i) for i in range(1,10)]
split_nums = [str(i) for i in range(1,47)]
grdi_splits = [str(i) for i in range(47,59)]
drugs=["AMP","AMC","AZM","CHL","CIP","CRO","FIS","FOX","GEN","NAL","SXT","TET","TIO"]
datasets=["public","grdi"]
dataset_paths = ['','grdi_']

rule all:
    input:
        expand("data/multi-mer/feat_ranks/{dataset}_1000_{drug}_"+kmer_length+"mer_feature_ranks.npy", dataset = datasets, drug = drugs)

rule clean:
    input:
        "data/genomes/raw/{id}.fasta"
    output:
        "data/genomes/clean/{id}.fasta"
    shell:
        "python src/clean.py {input} data/genomes/clean/"

rule grdi_clean:
    input:
        "data/grdi_genomes/raw/{gid}.fasta"
    output:
        "data/grdi_genomes/clean/{gid}.fasta"
    shell:
        "python src/clean.py {input} data/grdi_genomes/clean/"

rule count:
    input:
        "data/genomes/clean/{id}.fasta"
    output:
        temp("data/genomes/jellyfish_results"+kmer_length+"/{id}.jf")
    threads:
        2
    shell:
        "jellyfish count -C -m "+kmer_length+" -s 100M -t {threads} {input} -o {output}"

rule grdi_count:
    input:
        "data/grdi_genomes/clean/{gid}.fasta"
    output:
        temp("data/grdi_genomes/jellyfish_results"+kmer_length+"/{gid}.jf")
    threads:
        2
    shell:
        "jellyfish count -C -m "+kmer_length+" -s 100M -t {threads} {input} -o {output}"

rule dump:
    input:
        "data/genomes/jellyfish_results"+kmer_length+"/{id}.jf"
    output:
        "data/genomes/jellyfish_results"+kmer_length+"/{id}.fa"
    shell:
        "jellyfish dump {input} > {output}"

rule grdi_dump:
    input:
        "data/grdi_genomes/jellyfish_results"+kmer_length+"/{gid}.jf"
    output:
        "data/grdi_genomes/jellyfish_results"+kmer_length+"/{gid}.fa"
    shell:
        "jellyfish dump {input} > {output}"

rule union:
    # runs 5 times for ncbi, 4 for grdi
    input:
        expand("data/genomes/jellyfish_results"+kmer_length+"/{id}.fa", id=ids)
    output:
        "data/genomes/top_feats/all_"+kmer_length+"mers{set_num}.npy"
    shell:
        "python src/multi-mer/find_master.py "+kmer_length+" {output} {set_num}"

rule merge_masters:
    # runs once per dataset
    input:
        expand("data/genomes/top_feats/all_"+kmer_length+"mers{set_num}.npy", set_num = set_nums)
    output:
        "data/genomes/top_feats/{dataset_path}all_"+kmer_length+"mers.npy"
    shell:
        "python src/multi-mer/find_all_feats.py "+kmer_length+" {output} {dataset_path}"

rule matrix:
# runs 46 times for ncbi, 12 times for grdi
    input:
        expand("data/genomes/top_feats/{dataset_path}all_"+kmer_length+"mers.npy", dataset_path = dataset_paths)
    output:
        "data/multi-mer/splits/"+kmer_length+"mer_matrix{split}.npy"
    run:
        import os,sys
        import numpy as np

        if not os.path.exists(os.path.abspath(os.path.curdir)+"/data/multi-mer/genome_names.npy"):
            genomes = ([files for r,d,files in os.walk("data/genomes/jellyfish_results{}/".format(kmer_length))][0])
            np.save("data/multi-mer/genome_names.npy", genomes)
        if not os.path.exists(os.path.abspath(os.path.curdir)+"/data/multi-mer/genome_names.npy"):
            genomes = ([files for r,d,files in os.walk("data/grdi_genomes/jellyfish_results{}/".format(kmer_length))][0])
            np.save("data/multi-mer/grdi_genome_names.npy", genomes)


        shell("sbatch -c 16 --mem 700G --partition NMLResearch --wrap='python src/multi-mer/multi_mer_matrix.py {input} {wildcards.split} {kmer_length}'")

rule merge:
    input:
        expand("data/multi-mer/splits/"+kmer_length+"mer_matrix{split}.npy", split = split_nums)
    output:
        "data/multi-mer/"+kmer_length+"mer_matrix.npy",
        "data/multi-mer/"+kmer_length+"mer_rows.npy",
        "data/multi-mer/"+kmer_length+"mer_cols.npy"
    shell:
        "python src/multi-mer/matrix_merge.py {kmer_length} public"

rule grdi_merge:
    input:
        expand("data/multi-mer/splits/"+kmer_length+"mer_matrix{split}.npy", split = grdi_splits)
    output:
        "data/multi-mer/grdi_"+kmer_length+"mer_matrix.npy",
        "data/multi-mer/grdi_"+kmer_length+"mer_rows.npy",
        "data/multi-mer/grdi_"+kmer_length+"mer_cols.npy"
    shell:
        "python src/multi-mer/matrix_merge.py {kmer_length} grdi"

rule model:
    input:
        "data/multi-mer/grdi_"+kmer_length+"mer_matrix.npy",
        "data/multi-mer/"+kmer_length+"mer_matrix.npy"
    output:
        "data/multi-mer/feat_ranks/{dataset}_1000_{drug}_"+kmer_length+"mer_feature_ranks.npy"
    run:
        if wildcards.dataset == 'grdi' and wildcards.drug == 'FIS':
            shell("touch {output}")
        else:
            shell("sbatch -c 144 --mem 1007G --partition NMLResearch --wrap='python src/multi-mer/multi_mer_model.py {drug} 144 {dataset} 31 1000'")
"""
rule next:
    input:
        expand("data/multi-mer/feat_ranks/{dataset}_1000_{drug}_"+kmer_length+"mer_feature_ranks.npy", dataset = datasets, drug = drugs)
"""
