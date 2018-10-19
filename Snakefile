#################################################################

# Location of the raw genomes
#RAW_GENOMES_PATH = "/mnt/moria/rylan/phenores_data/raw/genomes/"
#RAW_GENOMES_PATH = "genomes/raw/"
RAW_GENOMES_PATH = "test/genomes/"

# Kmer length that you want to count
KMER_SIZE = 11

# Data type of the resulting kmer matrix. Use uint8 if counts are
# all under 256. Else use uint16 (kmer counts under 65536)
MATRIX_DTYPE = 'uint8'

#################################################################

ids, = glob_wildcards(RAW_GENOMES_PATH+"{id}.fasta")

rule all:
  input:
    "touchfile.txt"

rule clean:
  input:
    RAW_GENOMES_PATH+"{id}.fasta"
  output:
    "genomes/clean/{id}.fasta"
  run:
    shell("python scripts/clean.py {input} genomes/clean/")

rule count:
  input:
    "genomes/clean/{id}.fasta"
  output:
    temp("jellyfish_results/{id}.jf")
  threads:
    2
  shell:
    "jellyfish count -C -m {KMER_SIZE} -s 100M -t {threads} {input} -o {output}"

rule dump:
    input:
        "jellyfish_results/{id}.jf"
    output:
        "jellyfish_results/{id}.fa"
    shell:
        "jellyfish dump {input} > {output}"

rule matrix:
    input:
        expand("jellyfish_results/{id}.fa", id=ids)
    output:
        touch("touchfile.txt")
    shell:
        "python scripts/parallel_matrix.py {KMER_SIZE} {MATRIX_DTYPE} jellyfish_results/ unfiltered/"
