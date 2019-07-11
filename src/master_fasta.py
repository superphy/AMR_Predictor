"""
This script merges all fastas in data/genomes/clean and data/grdi_genomes/clean
into a single master fasta with file names appended to contig headers
"""

import numpy as np
import os, sys
from Bio import Seq, SeqIO

total_contigs = 0

def new_contig(fasta_path):
    for record in SeqIO.parse(fasta_path, "fasta"):
        contig_seq = record.seq
        contig_seq = contig_seq._get_seq_str_and_check_alphabet(contig_seq)

        contig_header = record.id
        file_name = fasta_path.split('/')[-1].split('.')[0]
        contig_header = ">{}_{}".format(file_name, contig_header)

        with open("data/master.fasta",'a') as master:
            global total_contigs
            total_contigs += 1
            master.write(contig_header)
            master.write("\n")
            master.write(contig_seq)
            master.write("\n")

    return 0

if __name__ == "__main__":

    for dataset in ['ncbi', 'grdi']:

        if(dataset == 'ncbi'):
            path = 'data/genomes/clean'
        else:
            path = 'data/grdi_genomes/clean'

        fastas = [files for r,d,files in os.walk(path)][0]
        fastas = [path+'/'+i for i in fastas]

        print("found {} samples in the {} dataset".format(len(fastas),dataset))

        for fasta in fastas:
            new_contig(fasta)

        print("completed {}".format(dataset))
    print("Total: {}".format(total_contigs))
