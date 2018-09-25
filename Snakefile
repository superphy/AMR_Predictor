
#################################################################

# Location of the raw genomes
#RAW_GENOMES_PATH = "/mnt/moria/rylan/phenores_data/raw/genomes/"
RAW_GENOMES_PATH = "genomes/raw/"

# Location of the MIC data file (excel spreadsheet)
#MIC_DATA_FILE = "amr_data/no_ecoli_GenotypicAMR_Master.xlsx" # location of MIC data file
#MIC_DATA_FILE = "amr_data/GRDI_AMR_Master.xlsx"
MIC_DATA_FILE = "amr_data/no_coli_GenotypicAMR_Master.tsv"

# The number of input genomes. The number of rows must match the
# nubmer of rows in the MIC data file. The names of the genomes
# must also be consistent, but need not be in the same order.
NUM_INPUT_FILES = 2260
#NUM_INPUT_FILES = 7961

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

rule kmer_count:
  input:
    "genomes/clean/{id}.fasta"
  output:
    temp("jellyfish_results/{id}.jf")
  threads:
    2
  shell:
    "jellyfish count -m {KMER_SIZE} -s 100M -t {threads} {input} -o {output}"

rule fa_dump:
  input:
    "jellyfish_results/{id}.jf"
  output:
    "jellyfish_results/{id}.fa"
  shell:
    "jellyfish dump {input} > {output}"

rule make_matrix:
  input:
    expand("jellyfish_results/{id}.fa", id=ids)
  output:
    touch("touchfile.txt")
  params:
    class_labels="amr_data/class_ranges.yaml"
  run:
    shell("python scripts/parallel_matrix.py {NUM_INPUT_FILES} {KMER_SIZE} {MATRIX_DTYPE} jellyfish_results/ unfiltered/")
    shell("python scripts/convert_dict.py")
    shell("python scripts/bin_mics.py {MIC_DATA_FILE}")
    shell("python scripts/filter.py")
    shell("python scripts/amr_prep.py")
