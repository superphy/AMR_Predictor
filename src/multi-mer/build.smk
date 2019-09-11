ids, = glob_wildcards("data/genomes/raw/{id}.fasta")

kmer_length = '31'
set_nums = = ['1','2','3','4','5']

rule all:
    input:
        "data/genomes/top_feats/1000_"+kmer_length+"mers.npy"

rule clean:
    input:
        "data/genomes/raw/{id}.fasta"
    output:
        "data/genomes/clean/{id}.fasta"
    shell:
        "python src/clean.py {input} data/genomes/clean/"

rule count:
    input:
        "data/genomes/clean/{id}.fasta"
    output:
        temp("data/genomes/jellyfish_results"+kmer_length+"/{id}.jf")
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

rule union:
    # runs 5 times
    input:
        expand("data/genomes/jellyfish_results"+kmer_length+"/{id}.fa", id=ids)
    output:
        "data/genomes/top_feats/1000_"+kmer_length+"mers{set_num}.npy"
    shell:
        "python src/multi-mer/find_master.py "+kmer_length+" {output} {set_num}"

rule master:
    # runs once
    input:
        expand("data/genomes/top_feats/1000_"+kmer_length+"mers{set_num}.npy", set_num = set_nums)
    output:
        "data/genomes/top_feats/1000_"+kmer_length+"mers.npy"
    shell:
        "python src/multi-mer/find_top_1000.py "+kmer_length+" {output}"
